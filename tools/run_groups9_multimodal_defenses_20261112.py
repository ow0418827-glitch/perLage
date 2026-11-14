"""环境二（特征文字 + 独立模糊照片）面对二十六种/十四行图像防御的完整检索评估与独立复算。

严格仅评估共享长期记忆检索排名：
- 不触发购物、商城服务或外部策略大模型；
- 不重新训练、不重新选图、不更改查询图片（查询图始终保持原样）；
- 对共享记忆库中的全部图片（180 条正常商品 + 9 条原图 + 9 条修改图，共 207 张唯一图片）
  在每种防御条件下施加完全相同的防御变换；
- 保留全部 26 种防御设置（含扩散起点 99/199/399 的全部 5 个随机种子 1001..1005），汇总为 14 行；
- `verify` 在全新子进程与全新编码器实例中，独立重新计算全部 3,538,080 个候选分数与 18,720 次全库排名。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
if str(PROJECT / "tools") not in sys.path:
    sys.path.insert(0, str(PROJECT / "tools"))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from src.experiments.defenses import Defender, conditions, table_groups
from src.experiments.paired_metrics import paired_rank_metrics
from src.experiments.query_contract import verify_query_input
from src.experiments.shared_evaluation import make_eval_encoder
from src.visual.audit import digest, freeze_json, read_json, save_json
from src.visual.image_utils import sha256_file
from src.visual.visual_retrieval import (
    canonical_digest,
    detect_index_text_policy,
    encoder_image_decoder_signature,
    memory_index_text,
    resolve_image,
    unit,
)
from src.visual.encoders.joint import validated_image_weight


DEFAULT_ORIGIN_PLAN = PROJECT / "output/groups9_multimodal_20261110/EXPERIMENT_PLAN.json"
EXPECTED_ORIGIN_PLAN_SHA256 = "8821a4e55e26c7edac4c1ddfe8058ae0136bcac9f24c4727fc0c2c890d2010b3"
DEFAULT_ORIGIN_RETRIEVAL_DIR = PROJECT / "output/groups9_multimodal_20261110"
DEFAULT_RUN = PROJECT / "output/groups9_multimodal_defenses_20261112"

METHODS = ("lucid", "perlage_no_pixel")
METHOD_NAMES = {"lucid": "原始像素方法", "perlage_no_pixel": "无像素感知方法"}
BANK_NAMES = ("clean", "modified")
BANK_LABELS = {"clean": "原图库", "modified": "修改库"}

READ_ONLY_DIRS = (
    PROJECT / "output/experiment_base_groups9_20261109",
    PROJECT / "output/lucid_groups9_20261109",
    PROJECT / "output/perlage_no_pixel_groups9_20261109",
    PROJECT / "output/lucid_groups9_clean_shopping_20261110",
    PROJECT / "output/groups9_multimodal_20261110",
    PROJECT / "output/groups9_vague_multimodal_20261111",
    PROJECT / "output/groups9_target_image_20261111",
)

STOP_MARKER_NAME = "STOP_SHOPPING_REQUESTED.json"
SCHEDULE_NAME = "SCHEDULED_INPUTS.json"
FAST_NON_DIFFUSION_CONDITIONS = (
    "none",
    "blur1",
    "blur2",
    "jpeg75",
    "jpeg50",
    "resize512",
    "vae512",
)
DIFFUSION_CONDITIONS = tuple(
    f"diff512_t{t}_s{s}" for t in (99, 199, 399) for s in range(1001, 1006)
)


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def lock(path: Path, *, create: bool = True):
    """文件互斥锁，防止相同输出目录并行写入或重复执行。"""
    path = Path(path)
    if create:
        path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b" if create else "r+b")
    if create and handle.tell() == 0:
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise RuntimeError("相同实验已有活动进程，拒绝并行重复执行") from None
    return handle


def status(run: Path, state: str, **extra) -> None:
    value = dict(status=state, updated_at=utcnow(), pid=os.getpid(), **extra)
    save_json(run / "RUN_STATUS.json", value)
    (run / "RUN_STATUS.md").write_text(
        "# 环境二防御检索状态\n\n" + "\n".join(f"- {k}: {v}" for k, v in value.items()) + "\n",
        encoding="utf-8",
    )


def check_file(path: str | Path, expected: str) -> str:
    resolved = Path(path).resolve()
    actual = sha256_file(resolved)
    if actual != expected:
        raise ValueError(f"冻结文件摘要改变：{resolved}（期望 {expected}，实际 {actual}）")
    return actual


def assert_writable_run_dir(run: Path) -> Path:
    run = Path(run).resolve()
    for protected in READ_ONLY_DIRS:
        p = protected.resolve()
        if run == p or run.is_relative_to(p):
            raise ValueError(f"禁止写入只读历史实验目录：{run}")
    for external in (PROJECT.parent / "lage", PROJECT.parent / "minja"):
        ep = external.resolve()
        if run == ep or run.is_relative_to(ep):
            raise ValueError(f"禁止写入只读外部项目目录：{run}")
    return run


def assert_no_shopping(run: Path) -> dict:
    marker_path = Path(run) / STOP_MARKER_NAME
    if not marker_path.is_file():
        raise RuntimeError("缺少仅检索停止购物标记文件 STOP_SHOPPING_REQUESTED.json")
    marker = read_json(marker_path)
    if (
        marker.get("shopping_allowed") is not False
        or marker.get("external_model_calls_allowed") is not False
        or marker.get("generation_enabled") is not False
        or marker.get("retrieval_only") is not True
    ):
        raise RuntimeError("实验被禁止购物与外部模型调用，停止标记校验失败")
    return marker


class MemoizedEncoder:
    """单进程内按 (text, image_sha256, image_weight) 记忆化融合向量，不跨进程读取外部缓存。"""

    def __init__(self, encoder):
        self.encoder = encoder
        self.snapshot = encoder.snapshot
        self.supports_image_weight = getattr(encoder, "supports_image_weight", True)
        self.cache: dict[tuple[str, str, float], np.ndarray] = {}

    def encode_joint(self, text: str, image: str | Path, image_weight: float = 0.5, *, _verified_sha256: str | None = None) -> np.ndarray:
        weight = validated_image_weight(image_weight)
        img_sha = _verified_sha256 if _verified_sha256 is not None else sha256_file(image)
        key = (str(text), img_sha, weight)
        if key not in self.cache:
            if weight == 0.5:
                vec = self.encoder.encode_joint(text, image)
            else:
                vec = self.encoder.encode_joint(text, image, image_weight=weight)
            self.cache[key] = unit(vec)
        return self.cache[key]


def fresh_encoder(plan: dict, device: str = "cuda", encoder_factory=None):
    if encoder_factory is not None:
        raw = encoder_factory()
        return raw, MemoizedEncoder(raw)
    raw = make_eval_encoder(device, plan["encoder_config"])
    surrogate = raw.snapshot.get("surrogate", {})
    if surrogate.get("device") != device:
        raw.close()
        raise ValueError(f"实际运行设备不符：期望 {device}，实际 {surrogate.get('device')}")
    if device == "cuda":
        loading = surrogate.get("loading", {})
        if loading.get("actual_activation_modules") != {"QuickGELU": 24, "GELU": 0}:
            raw.close()
            raise ValueError(f"QuickGELU 快速激活核验失败：{loading.get('actual_activation_modules')}")
    expected_snapshot_sha = plan.get("expected_encoder_snapshot_sha256")
    if expected_snapshot_sha is not None:
        if digest(raw.snapshot) != expected_snapshot_sha:
            raw.close()
            raise ValueError(
                f"评估编码器快照与环境二冻结配置不符：期望 {expected_snapshot_sha}，实际 {digest(raw.snapshot)}"
            )
    return raw, MemoizedEncoder(raw)


def _baseline_encoder_snapshot_sha(origin_retrieval_dir: Path | None, origin: dict) -> str | None:
    if origin_retrieval_dir is None:
        return None
    first_group = origin["methods"][METHODS[0]]["groups"][0]["key"]
    sample_file = Path(origin_retrieval_dir) / "methods" / METHODS[0] / "retrieval" / first_group / "clean.json"
    if not sample_file.is_file():
        return None
    sample = read_json(sample_file)
    return digest(sample["protocol"]["encoder"])


def prepare(
    run: Path = DEFAULT_RUN,
    *,
    origin_plan: Path = DEFAULT_ORIGIN_PLAN,
    origin_retrieval_dir: Path | None = DEFAULT_ORIGIN_RETRIEVAL_DIR,
    expected_origin_sha256: str | None = EXPECTED_ORIGIN_PLAN_SHA256,
) -> dict:
    """冻结环境二全部输入、207 张唯一库图片摘要与 26 种防御设置，写入停止购物标记。"""
    run = assert_writable_run_dir(run)
    run.mkdir(parents=True, exist_ok=True)

    origin_plan = Path(origin_plan).resolve()
    origin_sha = sha256_file(origin_plan)
    if expected_origin_sha256 is not None and origin_sha != expected_origin_sha256:
        raise ValueError(f"环境二原实验计划摘要改变：{origin_sha} != {expected_origin_sha256}")

    origin = read_json(origin_plan)
    if tuple(origin.get("expected_methods", ())) != METHODS:
        raise ValueError(f"原实验方法列表不符：{origin.get('expected_methods')}")
    if origin.get("query_mode") != "joint" or origin.get("query_image_weight") != 0.5 or origin.get("key_image_weight") != 0.5:
        raise ValueError("环境二必须为图文联合查询与图文各半记忆键")
    if tuple(origin.get("retrieval_banks", ())) != BANK_NAMES:
        raise ValueError("检索库范围必须同时包含 clean 与 modified")

    check_file(origin["pool"], origin["pool_sha256"])
    check_file(origin["encoder_config"], origin["encoder_config_sha256"])
    pool_count = int(origin["pool_count"])

    defense_specs = conditions()
    rows = table_groups()
    row_labels = [lbl for _, lbls in rows for lbl in lbls]
    if len(defense_specs) != 26 or len(rows) != 14 or row_labels != [s["label"] for s in defense_specs]:
        raise ValueError("固定防御面板定义异常")

    unique_assets: dict[str, dict] = {}
    checked_query_images: dict[str, str] = {}
    methods_payload: dict[str, dict] = {}

    for method in METHODS:
        mdata = origin["methods"][method]
        groups_out = []
        for group in mdata["groups"]:
            for pkey, skey in (
                ("frozen_file", "frozen_file_sha256"),
                ("frozen_image", "frozen_image_sha256"),
                ("manifest", "manifest_sha256"),
                ("text_source", "text_source_sha256"),
                ("image_review_path", "image_review_sha256"),
                ("previous_policy_bank", "previous_policy_bank_sha256"),
            ):
                if pkey in group and skey in group:
                    check_file(group[pkey], group[skey])

            queries = group["queries"]
            if not queries:
                raise ValueError(f"{method} {group['key']} 请求列表为空")
            for q in queries:
                verify_query_input(q)
                if q.get("query_mode") != "joint":
                    raise ValueError(f"环境二查询必须为图文联合模式：{q['task_id']}")
                qi = str(Path(q["query_image"]).resolve())
                if qi in checked_query_images:
                    if checked_query_images[qi] != q["query_image_sha256"]:
                        raise ValueError(f"查询图片摘要冲突：{qi}")
                else:
                    checked_query_images[qi] = check_file(qi, q["query_image_sha256"])

            banks_out = {}
            for bank_name in BANK_NAMES:
                bound = group["banks"][bank_name]
                bank_path = Path(bound["path"]).resolve()
                check_file(bank_path, bound["sha256"])
                bank = read_json(bank_path)
                if len(bank) != pool_count or len({e["Id"] for e in bank}) != pool_count:
                    raise ValueError(f"共享库规模或唯一性不符：{bank_path}")
                index_policy = detect_index_text_policy(bank)
                if index_policy != "product_feature":
                    raise ValueError(f"记忆文本键策略须为 product_feature，实际为 {index_policy}")
                source_entry = next((e for e in bank if e["Id"] == group["source_memory_id"]), None)
                if source_entry is None or source_entry["FinalASIN"] != group["source"]["id"]:
                    raise ValueError(f"源记忆商品映射校验失败：{method} {group['key']} {bank_name}")

                for entry in bank:
                    if entry.get("RecordKind") == "simulated_gallery" and (
                        entry.get("Completed") is True or entry.get("Actions")
                    ):
                        raise ValueError("发现模拟图库参考伪造已完成购买轨迹")
                    _ = memory_index_text(entry, index_policy)
                    img_path = resolve_image(entry, bank_path)
                    if img_path is None:
                        raise ValueError(f"记忆缺少有效图片：{entry['Id']}")
                    asset = entry.get("RetrievalImage") or entry["PrimaryImage"]
                    asset_sha = asset["SHA256"]
                    resolved_str = str(img_path.resolve())
                    if asset_sha not in unique_assets:
                        unique_assets[asset_sha] = dict(
                            sha256=asset_sha,
                            path=resolved_str,
                            aliases=[resolved_str],
                        )
                    elif resolved_str not in unique_assets[asset_sha]["aliases"]:
                        unique_assets[asset_sha]["aliases"].append(resolved_str)

                banks_out[bank_name] = dict(path=str(bank_path), sha256=bound["sha256"])

            groups_out.append(
                dict(
                    key=group["key"],
                    name=group["name"],
                    source=group["source"],
                    target=group["target"],
                    source_memory_id=group["source_memory_id"],
                    frozen_image=str(Path(group["frozen_image"]).resolve()),
                    frozen_image_sha256=group["frozen_image_sha256"],
                    banks=banks_out,
                    queries=[
                        dict(
                            task_id=q["task_id"],
                            instruction=q["instruction"],
                            query_mode=q["query_mode"],
                            query_image=str(Path(q["query_image"]).resolve()),
                            query_image_sha256=q["query_image_sha256"],
                        )
                        for q in queries
                    ],
                )
            )
        methods_payload[method] = dict(
            historical_method=mdata.get("historical_method", method),
            groups=groups_out,
        )

    num_groups = len(methods_payload[METHODS[0]]["groups"])
    queries_per_group = len(methods_payload[METHODS[0]]["groups"][0]["queries"])
    for m in METHODS:
        if len(methods_payload[m]["groups"]) != num_groups:
            raise ValueError("不同方法的分组数量不一致")
        for g in methods_payload[m]["groups"]:
            if len(g["queries"]) != queries_per_group:
                raise ValueError("不同组的正式请求数量不一致")

    unique_list = []
    for sha in sorted(unique_assets):
        rec = unique_assets[sha]
        aliases = sorted(rec["aliases"])
        unique_list.append(dict(sha256=sha, path=aliases[0], aliases=aliases))

    baseline_dir_str = None
    baseline_files: dict[str, str] = {}
    if origin_retrieval_dir is not None and Path(origin_retrieval_dir).is_dir():
        bdir = Path(origin_retrieval_dir).resolve()
        baseline_dir_str = str(bdir)
        for m in METHODS:
            for g in methods_payload[m]["groups"]:
                for bname in BANK_NAMES:
                    bf = bdir / "methods" / m / "retrieval" / g["key"] / f"{bname}.json"
                    if bf.is_file():
                        baseline_files[f"{m}:{g['key']}:{bname}"] = sha256_file(bf)

    expected_rows = len(METHODS) * num_groups * len(BANK_NAMES) * len(defense_specs)
    expected_rankings = expected_rows * queries_per_group
    expected_score_comparisons = expected_rankings * pool_count

    stop_marker = dict(
        schema_version=1,
        shopping_allowed=False,
        external_model_calls_allowed=False,
        generation_enabled=False,
        retrieval_only=True,
        environment="env2_feature_text_plus_vague_photo",
        reason="用户任务仅评估环境二面对防御时的检索效果，严禁调用商城、购买或外部策略模型",
    )
    freeze_json(run / STOP_MARKER_NAME, stop_marker)

    plan = dict(
        schema_version=1,
        environment="env2_feature_text_plus_vague_photo",
        origin_plan=str(origin_plan),
        origin_plan_sha256=origin_sha,
        is_frozen_env2_default=(origin_sha == EXPECTED_ORIGIN_PLAN_SHA256),
        origin_retrieval_dir=baseline_dir_str,
        baseline_files=baseline_files,
        expected_encoder_snapshot_sha256=_baseline_encoder_snapshot_sha(
            Path(baseline_dir_str) if baseline_dir_str else None, origin
        ),
        pool=str(Path(origin["pool"]).resolve()),
        pool_sha256=origin["pool_sha256"],
        pool_count=pool_count,
        encoder_config=str(Path(origin["encoder_config"]).resolve()),
        encoder_config_sha256=origin["encoder_config_sha256"],
        key_mode="joint",
        query_mode="joint",
        index_text="product_feature",
        key_image_weight=0.5,
        query_image_weight=0.5,
        query_image_defended=False,
        all_memory_images_equally_defended=True,
        shopping_enabled=False,
        generation_enabled=False,
        external_model_calls_enabled=False,
        expected_methods=list(METHODS),
        num_groups=num_groups,
        queries_per_group=queries_per_group,
        conditions=defense_specs,
        table_groups=[dict(title=title, labels=labels) for title, labels in rows],
        unique_images=unique_list,
        expected_rows=expected_rows,
        expected_rankings=expected_rankings,
        expected_score_comparisons=expected_score_comparisons,
        defenses_implementation_sha256=sha256_file(PROJECT / "src/experiments/defenses.py"),
        methods=methods_payload,
        caveats=list(origin.get("caveats", []))
        + [
            "扩散净化起点 99/199/399 各完整保留 5 个随机种子（1001..1005），同请求不同种子并非独立需求",
            "前三及前五仅反映检索阶段排序位置，不代表实际购物或误购率；本轮严禁购物与外部模型调用",
        ],
    )
    freeze_json(run / SCHEDULE_NAME, plan)
    plan_sha = sha256_file(run / SCHEDULE_NAME)

    freeze_json(
        run / "PREPARATION_COMPLETED.json",
        dict(
            status="passed",
            plan_sha256=plan_sha,
            stop_marker_sha256=sha256_file(run / STOP_MARKER_NAME),
            methods=len(METHODS),
            groups=num_groups,
            queries_per_group=queries_per_group,
            conditions=len(defense_specs),
            table_rows=len(rows),
            unique_bank_images=len(unique_list),
            expected_rows=expected_rows,
            expected_rankings=expected_rankings,
            expected_score_comparisons=expected_score_comparisons,
            shopping_calls=0,
            generation_calls=0,
        ),
    )
    status(
        run,
        "prepared",
        unique_images=len(unique_list),
        expected_rows=expected_rows,
        expected_rankings=expected_rankings,
        expected_score_comparisons=expected_score_comparisons,
    )
    return plan


def validate_plan(run: Path) -> tuple[dict, str]:
    run = Path(run).resolve()
    assert_no_shopping(run)
    plan_file = run / SCHEDULE_NAME
    if not plan_file.is_file():
        raise FileNotFoundError(f"缺少冻结调度计划：{plan_file}")
    plan = read_json(plan_file)
    plan_sha = sha256_file(plan_file)
    if (
        plan.get("shopping_enabled") is not False
        or plan.get("generation_enabled") is not False
        or plan.get("external_model_calls_enabled") is not False
        or plan.get("query_image_defended") is not False
        or plan.get("all_memory_images_equally_defended") is not True
    ):
        raise ValueError("冻结计划违背仅检索或查询保持原图约定")
    check_file(plan["origin_plan"], plan["origin_plan_sha256"])
    check_file(plan["pool"], plan["pool_sha256"])
    check_file(plan["encoder_config"], plan["encoder_config_sha256"])
    if plan["conditions"] != conditions():
        raise ValueError("防御条件列表与标准 26 项面板不符")
    for item in plan["unique_images"]:
        check_file(item["path"], item["sha256"])
    checked_q: set[str] = set()
    for method in METHODS:
        for group in plan["methods"][method]["groups"]:
            check_file(group["frozen_image"], group["frozen_image_sha256"])
            for bank_name in BANK_NAMES:
                b = group["banks"][bank_name]
                check_file(b["path"], b["sha256"])
            for q in group["queries"]:
                verify_query_input(q)
                if q["query_image"] not in checked_q:
                    check_file(q["query_image"], q["query_image_sha256"])
                    checked_q.add(q["query_image"])
    return plan, plan_sha


def resolve_selected_conditions(
    plan: dict,
    selected: str | list[str] | tuple[str, ...] | None = None,
) -> tuple[list[dict], bool]:
    """解析并按固定 26 项面板顺序规范化防御条件子集，返回 (specs, is_full_panel)。"""
    all_specs = plan["conditions"]
    if selected is None:
        return list(all_specs), True
    if isinstance(selected, str):
        parts = [p.strip() for p in selected.split(",")]
    elif isinstance(selected, (bytes, bytearray)):
        raise ValueError("防御条件参数不能为字节串")
    else:
        parts = [str(p).strip() for p in selected]
    if not parts or any(not p for p in parts):
        raise ValueError(f"防御条件选择为空或包含空项：{selected!r}")
    seen: set[str] = set()
    for p in parts:
        if p in seen:
            raise ValueError(f"防御条件包含重复项：{p}")
        seen.add(p)
    valid_labels = {s["label"] for s in all_specs}
    unknown = [p for p in parts if p not in valid_labels]
    if unknown:
        raise ValueError(f"包含未知防御条件标签：{unknown}")
    ordered_specs = [s for s in all_specs if s["label"] in seen]
    return ordered_specs, len(ordered_specs) == len(all_specs)


def resolve_subset_name(
    specs: list[dict],
    is_full_panel: bool,
    subset_name: str | None = None,
) -> str | None:
    """为部分条件实验解析安全且确定性的子集目录名；全量 26 项时拒绝子集名。"""
    if is_full_panel:
        if subset_name is not None:
            raise ValueError("全量 26 项实验禁止指定 --subset-name")
        return None
    if subset_name is not None:
        if not isinstance(subset_name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", subset_name):
            raise ValueError(
                f"非法子集名称（仅允许 1..64 位英文数字下划线与短横线）：{subset_name!r}"
            )
        return subset_name
    labels = tuple(s["label"] for s in specs)
    if labels == FAST_NON_DIFFUSION_CONDITIONS:
        return "fast_non_diffusion"
    return f"conds_{digest(list(labels))[:12]}"


def _cleanup_orphan_defense_files(cond_dir: Path) -> None:
    if not cond_dir.is_dir():
        return
    for png in cond_dir.glob("*.png"):
        if not png.with_suffix(".json").is_file():
            png.unlink()
    for sidecar in cond_dir.glob("*.json"):
        if not sidecar.with_suffix(".png").is_file():
            sidecar.unlink()


def _validate_completed_condition(
    cond_marker: Path,
    spec: dict,
    plan_sha: str,
    expected_shas: set[str],
    *,
    require_sidecar: bool = True,
    expected_impl_sha: str | None = None,
) -> dict:
    doc = read_json(cond_marker)
    if doc.get("status") != "completed" or doc.get("plan_sha256") != plan_sha or doc.get("condition") != spec:
        raise ValueError(f"防御完成标记与计划不符：{cond_marker}")
    items = doc.get("items", {})
    if set(items) != expected_shas:
        raise ValueError(f"防御完成条目未完整覆盖所有唯一库图片：{cond_marker}")
    for orig_sha, entry in items.items():
        actual_def_sha = check_file(entry["defended_path"], entry["defended_sha256"])
        if spec["kind"] == "none":
            if actual_def_sha != orig_sha:
                raise ValueError(f"无防御条件返回了修改图片：{orig_sha}")
        else:
            sidecar_path = entry.get("sidecar_path")
            if not sidecar_path:
                if require_sidecar:
                    raise ValueError(f"缺少防御变换身份旁车文件：{entry['defended_path']}")
                continue
            if not Path(sidecar_path).is_file():
                raise ValueError(f"缺少防御变换身份旁车文件：{entry['defended_path']}")
            check_file(sidecar_path, entry["sidecar_sha256"])
            sidecar = read_json(sidecar_path)
            ident = sidecar.get("identity", {})
            if (
                sidecar.get("output_sha256") != actual_def_sha
                or ident.get("input_sha256") != orig_sha
                or ident.get("condition") != spec
                or (expected_impl_sha is not None and ident.get("implementation_sha256") != expected_impl_sha)
            ):
                raise ValueError(f"防御旁车文件与实际图片或条件不符：{sidecar_path}")
    return items


def defend(
    run: Path,
    device: str = "cuda",
    defense_factory=None,
    selected_conditions: str | list[str] | tuple[str, ...] | None = None,
) -> dict:
    """对全部 207 张唯一库图片施加选定（默认全部 26 种）统一防御变换并写入不可篡改索引。"""
    run = assert_writable_run_dir(run)
    plan, plan_sha = validate_plan(run)
    selected_specs, is_full_panel = resolve_selected_conditions(plan, selected_conditions)
    require_sidecar = defense_factory is None
    expected_impl_sha = plan["defenses_implementation_sha256"] if require_sidecar else None
    if require_sidecar:
        check_file(PROJECT / "src/experiments/defenses.py", plan["defenses_implementation_sha256"])

    unique_images = plan["unique_images"]
    expected_shas = {item["sha256"] for item in unique_images}
    defended_root = run / "defended"
    defended_root.mkdir(parents=True, exist_ok=True)

    defender = defense_factory(defended_root, device) if defense_factory else Defender(defended_root, device)
    selected_conditions_out: dict[str, dict] = {}
    try:
        total = len(selected_specs)
        for idx, spec in enumerate(selected_specs, 1):
            label = spec["label"]
            cond_marker = defended_root / f"COMPLETED_{label}.json"
            if cond_marker.is_file():
                items = _validate_completed_condition(
                    cond_marker,
                    spec,
                    plan_sha,
                    expected_shas,
                    require_sidecar=require_sidecar,
                    expected_impl_sha=expected_impl_sha,
                )
                selected_conditions_out[label] = items
                print(f"[defend] ({idx}/{total}) {label}: already completed & verified", flush=True)
                continue

            status(
                run,
                "running",
                stage="defend",
                is_subset=not is_full_panel,
                condition=label,
                condition_index=idx,
                conditions_total=total,
            )
            _cleanup_orphan_defense_files(defended_root / label)
            t0 = time.perf_counter()
            items = {}
            for img in unique_images:
                orig_path = Path(img["path"]).resolve()
                orig_sha = img["sha256"]
                out_path = Path(defender(orig_path, spec)).resolve()
                out_sha = sha256_file(out_path)
                if spec["kind"] == "none":
                    if out_sha != orig_sha:
                        raise ValueError(f"none 条件图片内容改变：{orig_path}")
                    items[orig_sha] = dict(
                        orig_path=str(orig_path),
                        orig_sha256=orig_sha,
                        defended_path=str(out_path),
                        defended_sha256=out_sha,
                        sidecar_path=None,
                        sidecar_sha256=None,
                    )
                else:
                    sidecar = out_path.with_suffix(".json")
                    if not sidecar.is_file():
                        # 兼容极少数外部注入仅返回路径的测试双工；真实 Defender 必定生成旁车文件
                        if defense_factory is None:
                            raise ValueError(f"真实 Defender 未产出旁车身份文件：{out_path}")
                        sidecar_sha = None
                        sidecar_str = None
                    else:
                        sidecar_data = read_json(sidecar)
                        if sidecar_data.get("output_sha256") != out_sha:
                            raise ValueError(f"防御输出旁车摘要不一致：{sidecar}")
                        sidecar_sha = sha256_file(sidecar)
                        sidecar_str = str(sidecar.resolve())
                    items[orig_sha] = dict(
                        orig_path=str(orig_path),
                        orig_sha256=orig_sha,
                        defended_path=str(out_path),
                        defended_sha256=out_sha,
                        sidecar_path=sidecar_str,
                        sidecar_sha256=sidecar_sha,
                    )
            freeze_json(
                cond_marker,
                dict(
                    status="completed",
                    plan_sha256=plan_sha,
                    condition=spec,
                    image_count=len(items),
                    items=items,
                ),
            )
            selected_conditions_out[label] = items
            elapsed = time.perf_counter() - t0
            print(f"[defend] ({idx}/{total}) {label}: {len(items)} images done in {elapsed:.2f}s", flush=True)
    finally:
        if hasattr(defender, "purifier") and defender.purifier is not None:
            defender.purifier = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    all_completed = all((defended_root / f"COMPLETED_{s['label']}.json").is_file() for s in plan["conditions"])
    if all_completed:
        full_conditions: dict[str, dict] = {}
        for spec in plan["conditions"]:
            lbl = spec["label"]
            if lbl in selected_conditions_out:
                full_conditions[lbl] = selected_conditions_out[lbl]
            else:
                full_conditions[lbl] = _validate_completed_condition(
                    defended_root / f"COMPLETED_{lbl}.json",
                    spec,
                    plan_sha,
                    expected_shas,
                    require_sidecar=require_sidecar,
                    expected_impl_sha=expected_impl_sha,
                )
        index_payload = dict(
            schema_version=1,
            plan_sha256=plan_sha,
            unique_image_count=len(unique_images),
            conditions_count=len(plan["conditions"]),
            conditions=full_conditions,
        )
        index_file = defended_root / "INDEX.json"
        freeze_json(index_file, index_payload)
        freeze_json(
            run / "DEFENSE_COMPLETED.json",
            dict(
                status="completed",
                plan_sha256=plan_sha,
                conditions=len(plan["conditions"]),
                unique_images=len(unique_images),
                index_sha256=sha256_file(index_file),
                device=device,
            ),
        )
        status(run, "defended", conditions=len(plan["conditions"]), unique_images=len(unique_images))
        return index_payload if is_full_panel else dict(
            schema_version=1,
            is_subset=True,
            plan_sha256=plan_sha,
            unique_image_count=len(unique_images),
            conditions_count=len(selected_specs),
            conditions=selected_conditions_out,
        )

    completed_labels = [
        s["label"] for s in plan["conditions"] if (defended_root / f"COMPLETED_{s['label']}.json").is_file()
    ]
    status(
        run,
        "defended_subset",
        selected_conditions=[s["label"] for s in selected_specs],
        completed_conditions_count=len(completed_labels),
        total_conditions=len(plan["conditions"]),
        unique_images=len(unique_images),
    )
    return dict(
        schema_version=1,
        is_subset=True,
        plan_sha256=plan_sha,
        unique_image_count=len(unique_images),
        conditions_count=len(selected_specs),
        conditions=selected_conditions_out,
    )


def _load_and_verify_defended_index(
    run: Path,
    plan: dict,
    plan_sha: str,
    *,
    selected_specs: list[dict] | None = None,
    require_sidecar: bool = True,
) -> dict:
    expected_shas = {item["sha256"] for item in plan["unique_images"]}
    expected_impl_sha = plan["defenses_implementation_sha256"] if require_sidecar else None

    if selected_specs is not None and len(selected_specs) < len(plan["conditions"]):
        conditions_out: dict[str, dict] = {}
        for spec in selected_specs:
            lbl = spec["label"]
            cond_marker = run / "defended" / f"COMPLETED_{lbl}.json"
            if not cond_marker.is_file():
                raise FileNotFoundError(f"尚未完成条件 {lbl} 的图像防御阶段（缺少 {cond_marker}）")
            conditions_out[lbl] = _validate_completed_condition(
                cond_marker,
                spec,
                plan_sha,
                expected_shas,
                require_sidecar=require_sidecar,
                expected_impl_sha=expected_impl_sha,
            )
        return dict(
            schema_version=1,
            is_subset=True,
            plan_sha256=plan_sha,
            unique_image_count=len(expected_shas),
            conditions_count=len(selected_specs),
            conditions=conditions_out,
        )

    marker_file = run / "DEFENSE_COMPLETED.json"
    if not marker_file.is_file():
        raise FileNotFoundError("尚未完成图像防御阶段（缺少 DEFENSE_COMPLETED.json）")
    marker = read_json(marker_file)
    if marker.get("status") != "completed" or marker.get("plan_sha256") != plan_sha:
        raise ValueError("DEFENSE_COMPLETED.json 状态或计划绑定异常")
    index_file = run / "defended/INDEX.json"
    check_file(index_file, marker["index_sha256"])
    index_doc = read_json(index_file)
    if index_doc.get("plan_sha256") != plan_sha:
        raise ValueError("defended/INDEX.json 计划绑定异常")
    expected_labels = {s["label"]: s for s in plan["conditions"]}
    if set(index_doc.get("conditions", {})) != set(expected_labels):
        raise ValueError("defended/INDEX.json 未完整覆盖 26 种防御条件")
    for label, spec in expected_labels.items():
        items = index_doc["conditions"][label]
        if set(items) != expected_shas:
            raise ValueError(f"条件 {label} 图片索引缺失")
        for orig_sha, entry in items.items():
            actual_sha = check_file(entry["defended_path"], entry["defended_sha256"])
            if spec["kind"] == "none":
                if actual_sha != orig_sha:
                    raise ValueError(f"none 条件图片已被篡改：{orig_sha}")
            elif require_sidecar:
                sidecar_path = entry.get("sidecar_path")
                if not sidecar_path or not Path(sidecar_path).is_file():
                    raise ValueError(f"缺少防御旁车文件：{entry['defended_path']}")
                check_file(sidecar_path, entry["sidecar_sha256"])
                sc = read_json(sidecar_path)
                if (
                    sc.get("output_sha256") != actual_sha
                    or sc.get("identity", {}).get("input_sha256") != orig_sha
                    or sc.get("identity", {}).get("condition") != spec
                    or sc.get("identity", {}).get("implementation_sha256") != plan["defenses_implementation_sha256"]
                ):
                    raise ValueError(f"防御图片旁车元数据检验失败：{sidecar_path}")
    return index_doc


def _load_bank_records(plan: dict) -> dict[tuple[str, str, str], dict]:
    """独立解析并核验全部共享库 JSON，返回文本键与原图摘要记录。"""
    loaded: dict[tuple[str, str, str], dict] = {}
    pool_count = int(plan["pool_count"])
    for method in METHODS:
        for group in plan["methods"][method]["groups"]:
            for bank_name in BANK_NAMES:
                binfo = group["banks"][bank_name]
                bank_path = Path(binfo["path"]).resolve()
                check_file(bank_path, binfo["sha256"])
                bank = read_json(bank_path)
                if len(bank) != pool_count or len({e["Id"] for e in bank}) != pool_count:
                    raise ValueError(f"记忆库规模或唯一性不符：{bank_path}")
                index_policy = detect_index_text_policy(bank)
                records = []
                for idx, entry in enumerate(bank):
                    txt = memory_index_text(entry, index_policy)
                    asset = entry.get("RetrievalImage") or entry["PrimaryImage"]
                    records.append(
                        dict(
                            index=idx,
                            Id=entry["Id"],
                            text=txt,
                            text_sha256=hashlib.sha256(txt.encode("utf-8")).hexdigest(),
                            orig_sha256=asset["SHA256"],
                        )
                    )
                loaded[(method, group["key"], bank_name)] = dict(
                    bank_path=str(bank_path),
                    bank_sha256=binfo["sha256"],
                    entries_sha256=canonical_digest(bank),
                    index_policy=index_policy,
                    source_memory_id=group["source_memory_id"],
                    records=records,
                )
    return loaded


def _defended_bank_fingerprint(records: list[dict], cond_items: dict[str, dict]) -> str:
    payload = [
        dict(
            Id=r["Id"],
            orig_sha256=r["orig_sha256"],
            defended_sha256=cond_items[r["orig_sha256"]]["defended_sha256"],
            text_sha256=r["text_sha256"],
        )
        for r in records
    ]
    return canonical_digest(payload)


def _check_none_against_baseline(
    plan: dict,
    method: str,
    scores_arr: np.ndarray,
    method_json: dict,
) -> dict | None:
    bdir = plan.get("origin_retrieval_dir")
    if not bdir:
        return None
    bpath = Path(bdir)
    if not bpath.is_dir():
        return None
    max_diff = 0.0
    checked = 0
    groups = plan["methods"][method]["groups"]
    for g_idx, group in enumerate(groups):
        gkey = group["key"]
        for b_idx, bank_name in enumerate(BANK_NAMES):
            old_file = bpath / "methods" / method / "retrieval" / gkey / f"{bank_name}.json"
            if not old_file.is_file():
                continue
            key = f"{method}:{gkey}:{bank_name}"
            if key in plan.get("baseline_files", {}):
                check_file(old_file, plan["baseline_files"][key])
            old_doc = read_json(old_file)
            g_res = method_json["groups"][g_idx]["banks"][bank_name]
            for q_idx, old_q in enumerate(old_doc["rankings"]):
                cur_q = g_res["rankings"][q_idx]
                if cur_q["source_rank"] != old_q["source_rank"]:
                    raise ValueError(
                        f"none 条件源名次与环境二基线不符：{method} {gkey} {bank_name} {cur_q['task_id']} "
                        f"({cur_q['source_rank']} != {old_q['source_rank']})"
                    )
                row_scores = scores_arr[g_idx, b_idx, q_idx, :]
                for item in old_q["ranking"]:
                    cidx = item["index"]
                    diff = abs(float(row_scores[cidx]) - float(item["score"]))
                    if not math.isfinite(diff) or diff > 1e-6:
                        raise ValueError(
                            f"none 条件分数与环境二基线偏差过大：{method} {gkey} {bank_name} {cur_q['task_id']} diff={diff}"
                        )
                    max_diff = max(max_diff, diff)
                    checked += 1
    if plan.get("is_frozen_env2_default"):
        clean_top3 = sum(g["banks"]["clean"]["top3"] for g in method_json["groups"])
        mod_top3 = sum(g["banks"]["modified"]["top3"] for g in method_json["groups"])
        expected_mod = 91 if method == "lucid" else 81
        if clean_top3 != 91 or mod_top3 != expected_mod:
            raise ValueError(
                f"环境二无防御汇总与基线截图不符：{method} clean={clean_top3}/180 mod={mod_top3}/180"
            )
    return dict(checked_candidates=checked, max_score_diff=max_diff)


def _method_retrieval_paths(run: Path, method: str, label: str) -> tuple[Path, Path]:
    base = run / "methods" / method / "retrieval"
    return base / f"{label}.json", base / f"{label}_scores.npz"


def _compute_ranking_order(scores_row: np.ndarray) -> list[int]:
    return sorted(range(len(scores_row)), key=lambda i: (-float(scores_row[i]), i))


def _check_retrieval_artifact(
    json_path: Path,
    npz_path: Path,
    plan: dict,
    plan_sha: str,
    method: str,
    spec: dict,
    cond_items: dict[str, dict],
    bank_records: dict[tuple[str, str, str], dict],
) -> tuple[dict, np.ndarray]:
    doc = read_json(json_path)
    if (
        doc.get("plan_sha256") != plan_sha
        or doc.get("method") != method
        or doc.get("condition") != spec
        or doc.get("pool_count") != plan["pool_count"]
    ):
        raise ValueError(f"检索结果元信息或计划绑定不符：{json_path}")
    check_file(npz_path, doc["scores_sha256"])
    with np.load(npz_path) as data:
        scores = np.array(data["scores"], dtype=np.float32)
    expected_shape = (
        plan["num_groups"],
        len(BANK_NAMES),
        plan["queries_per_group"],
        plan["pool_count"],
    )
    if scores.shape != expected_shape or not np.isfinite(scores).all():
        raise ValueError(f"分数矩阵形状或数值非法：{npz_path} {scores.shape}")

    groups = plan["methods"][method]["groups"]
    if len(doc.get("groups", [])) != len(groups):
        raise ValueError(f"分组数量异常：{json_path}")
    for g_idx, (group, g_res) in enumerate(zip(groups, doc["groups"])):
        if g_res["group"] != group["key"] or g_res["source_memory_id"] != group["source_memory_id"]:
            raise ValueError(f"分组身份不一致：{json_path}")
        for b_idx, bank_name in enumerate(BANK_NAMES):
            brec = bank_records[(method, group["key"], bank_name)]
            bres = g_res["banks"][bank_name]
            expected_fp = _defended_bank_fingerprint(brec["records"], cond_items)
            if (
                bres["orig_bank_sha256"] != brec["bank_sha256"]
                or bres["defended_bank_fingerprint"] != expected_fp
            ):
                raise ValueError(f"防御库指纹不一致：{json_path} {group['key']} {bank_name}")
            if len(bres["rankings"]) != len(group["queries"]):
                raise ValueError(f"请求名次列表长度不符：{json_path}")
            records = brec["records"]
            src_idx = next(r["index"] for r in records if r["Id"] == group["source_memory_id"])
            for q_idx, (q, row) in enumerate(zip(group["queries"], bres["rankings"])):
                expected_qtext_sha = hashlib.sha256(q["instruction"].encode("utf-8")).hexdigest()
                if (
                    row["task_id"] != q["task_id"]
                    or row.get("query_text_sha256") != expected_qtext_sha
                    or row.get("query_image_sha256") != q["query_image_sha256"]
                ):
                    raise ValueError(f"请求身份或查询图文字摘要错位：{json_path}")
                row_scores = scores[g_idx, b_idx, q_idx, :]
                order = _compute_ranking_order(row_scores)
                actual_rank = order.index(src_idx) + 1
                actual_score = float(row_scores[src_idx])
                if row["source_rank"] != actual_rank or abs(row["source_score"] - actual_score) > 1e-7:
                    raise ValueError(f"源记忆名次或分数与分数矩阵不符：{json_path} {q['task_id']}")
                if [records[i]["Id"] for i in order[:3]] != row["selected_ids"]:
                    raise ValueError(f"前三召回序列与分数矩阵不符：{json_path}")
                if [records[i]["Id"] for i in order[:5]] != row["top5_ids"]:
                    raise ValueError(f"前五召回序列与分数矩阵不符：{json_path}")
            ranks = [r["source_rank"] for r in bres["rankings"]]
            if (
                bres["top1"] != sum(r <= 1 for r in ranks)
                or bres["top3"] != sum(r <= 3 for r in ranks)
                or bres["top5"] != sum(r <= 5 for r in ranks)
                or abs(bres["mean_rank"] - (sum(ranks) / len(ranks))) > 1e-7
            ):
                raise ValueError(f"汇总计数或平均名次与逐请求名次不符：{json_path}")
    return doc, scores


def retrieve(
    run: Path,
    device: str = "cuda",
    encoder_factory=None,
    selected_conditions: str | list[str] | tuple[str, ...] | None = None,
    subset_name: str | None = None,
) -> dict:
    """用全库图像防御索引执行环境二检索，保留原始 float32 分数；支持子集隔离标记。"""
    run = assert_writable_run_dir(run)
    plan, plan_sha = validate_plan(run)
    selected_specs, is_full_panel = resolve_selected_conditions(plan, selected_conditions)
    resolved_subset = resolve_subset_name(selected_specs, is_full_panel, subset_name)
    selected_labels = [s["label"] for s in selected_specs]

    index_doc = _load_and_verify_defended_index(
        run,
        plan,
        plan_sha,
        selected_specs=selected_specs,
        require_sidecar=(encoder_factory is None),
    )
    bank_records = _load_bank_records(plan)

    raw_encoder, encoder = fresh_encoder(plan, device=device, encoder_factory=encoder_factory)
    decoder_sig = encoder_image_decoder_signature(raw_encoder)
    model_sha = digest(raw_encoder.snapshot)

    try:
        # 预先编码全部正式查询（环境二下查询图始终不经过防御）
        query_vecs: dict[tuple[str, str], np.ndarray] = {}
        for method in METHODS:
            for group in plan["methods"][method]["groups"]:
                for q in group["queries"]:
                    qkey = (q["instruction"], q["query_image_sha256"])
                    if qkey not in query_vecs:
                        query_vecs[qkey] = encoder.encode_joint(
                            q["instruction"],
                            Path(q["query_image"]).resolve(),
                            image_weight=0.5,
                            _verified_sha256=q["query_image_sha256"],
                        )

        total_tasks = len(METHODS) * len(selected_specs)
        task_num = 0
        per_method_records: dict[str, dict] = {}
        for method in METHODS:
            groups = plan["methods"][method]["groups"]
            method_files: dict[str, dict] = {}
            for spec in selected_specs:
                task_num += 1
                label = spec["label"]
                json_path, npz_path = _method_retrieval_paths(run, method, label)
                cond_items = index_doc["conditions"][label]

                if json_path.is_file() and npz_path.is_file():
                    existing_doc, existing_scores = _check_retrieval_artifact(
                        json_path, npz_path, plan, plan_sha, method, spec, cond_items, bank_records
                    )
                    if existing_doc["protocol"]["model_sha256"] != model_sha:
                        raise ValueError(f"续跑编码器快照身份不一致：{json_path}")
                    method_files[label] = dict(
                        json=str(json_path),
                        json_sha256=sha256_file(json_path),
                        scores=str(npz_path),
                        scores_sha256=sha256_file(npz_path),
                    )
                    print(f"[retrieve] ({task_num}/{total_tasks}) {method} {label}: resumed & validated", flush=True)
                    continue

                status(
                    run,
                    "running",
                    stage="retrieve",
                    is_subset=not is_full_panel,
                    subset_name=resolved_subset,
                    method=method,
                    condition=label,
                    progress=f"{task_num}/{total_tasks}",
                )
                t0 = time.perf_counter()
                scores_arr = np.empty(
                    (len(groups), len(BANK_NAMES), plan["queries_per_group"], plan["pool_count"]),
                    dtype=np.float32,
                )
                group_outputs = []
                for g_idx, group in enumerate(groups):
                    gkey = group["key"]
                    banks_out = {}
                    for b_idx, bank_name in enumerate(BANK_NAMES):
                        brec = bank_records[(method, gkey, bank_name)]
                        records = brec["records"]
                        src_idx = next(r["index"] for r in records if r["Id"] == group["source_memory_id"])
                        mem_vecs = [
                            encoder.encode_joint(
                                r["text"],
                                Path(cond_items[r["orig_sha256"]]["defended_path"]).resolve(),
                                image_weight=0.5,
                                _verified_sha256=cond_items[r["orig_sha256"]]["defended_sha256"],
                            )
                            for r in records
                        ]
                        q_rows = []
                        for q_idx, q in enumerate(group["queries"]):
                            qv = query_vecs[(q["instruction"], q["query_image_sha256"])]
                            row_scores = np.asarray([float(mv @ qv) for mv in mem_vecs], dtype=np.float32)
                            if not np.isfinite(row_scores).all():
                                raise ValueError(f"检测到非有限检索分数：{method} {gkey} {label} {q['task_id']}")
                            scores_arr[g_idx, b_idx, q_idx, :] = row_scores
                            order = _compute_ranking_order(row_scores)
                            src_rank = order.index(src_idx) + 1
                            src_score = float(row_scores[src_idx])
                            q_rows.append(
                                dict(
                                    task_id=q["task_id"],
                                    query_text_sha256=hashlib.sha256(q["instruction"].encode("utf-8")).hexdigest(),
                                    query_image_sha256=q["query_image_sha256"],
                                    source_rank=src_rank,
                                    source_score=src_score,
                                    selected_ids=[records[i]["Id"] for i in order[:3]],
                                    top5_ids=[records[i]["Id"] for i in order[:5]],
                                )
                            )
                        ranks = [x["source_rank"] for x in q_rows]
                        banks_out[bank_name] = dict(
                            bank_path=brec["bank_path"],
                            orig_bank_sha256=brec["bank_sha256"],
                            entries_sha256=brec["entries_sha256"],
                            defended_bank_fingerprint=_defended_bank_fingerprint(records, cond_items),
                            query_count=len(q_rows),
                            top1=sum(r <= 1 for r in ranks),
                            top3=sum(r <= 3 for r in ranks),
                            top5=sum(r <= 5 for r in ranks),
                            mean_rank=float(sum(ranks) / len(ranks)),
                            rankings=q_rows,
                        )
                    group_outputs.append(
                        dict(
                            group=gkey,
                            name=group["name"],
                            source_memory_id=group["source_memory_id"],
                            banks=banks_out,
                        )
                    )

                npz_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_npz = npz_path.with_name(npz_path.name + f".{os.getpid()}.tmp.npz")
                try:
                    np.savez(tmp_npz, scores=scores_arr)
                    if npz_path.exists():
                        with np.load(npz_path) as existing_d:
                            if np.array_equal(existing_d["scores"], scores_arr):
                                tmp_npz.unlink()
                            else:
                                raise ValueError(f"拒绝覆盖已存在的不同分数文件：{npz_path}")
                    else:
                        tmp_npz.replace(npz_path)
                finally:
                    if tmp_npz.exists():
                        tmp_npz.unlink()

                scores_sha = sha256_file(npz_path)
                payload = dict(
                    schema_version=1,
                    plan_sha256=plan_sha,
                    method=method,
                    condition=spec,
                    pool_count=plan["pool_count"],
                    scores_file=str(npz_path),
                    scores_sha256=scores_sha,
                    scores_shape=list(scores_arr.shape),
                    protocol=dict(
                        encoder=raw_encoder.snapshot,
                        model_sha256=model_sha,
                        image_decoder=decoder_sig,
                        key_mode="joint",
                        query_mode="joint",
                        index_text="product_feature",
                        key_image_weight=0.5,
                        query_image_weight=0.5,
                        query_image_defended=False,
                        all_memory_images_equally_defended=True,
                    ),
                    groups=group_outputs,
                )
                if label == "none":
                    baseline_check = _check_none_against_baseline(plan, method, scores_arr, payload)
                    if baseline_check is not None:
                        payload["baseline_check_vs_20261110"] = baseline_check
                freeze_json(json_path, payload)
                method_files[label] = dict(
                    json=str(json_path),
                    json_sha256=sha256_file(json_path),
                    scores=str(npz_path),
                    scores_sha256=scores_sha,
                )
                elapsed = time.perf_counter() - t0
                mod_t3 = sum(g["banks"]["modified"]["top3"] for g in group_outputs)
                cln_t3 = sum(g["banks"]["clean"]["top3"] for g in group_outputs)
                print(
                    f"[retrieve] ({task_num}/{total_tasks}) {method} {label}: clean={cln_t3}, mod={mod_t3} ({elapsed:.2f}s)",
                    flush=True,
                )

            m_rankings = len(selected_specs) * len(groups) * len(BANK_NAMES) * plan["queries_per_group"]
            m_comparisons = m_rankings * plan["pool_count"]
            method_marker_payload = dict(
                status="completed",
                is_subset=not is_full_panel,
                subset_name=resolved_subset,
                selected_labels=selected_labels,
                plan_sha256=plan_sha,
                method=method,
                conditions=len(selected_specs),
                rankings=m_rankings,
                score_comparisons=m_comparisons,
                execution_device=device,
                model_sha256=model_sha,
                files=method_files,
            )
            marker_dest = (
                run / "methods" / method / "RETRIEVAL_COMPLETED.json"
                if is_full_panel
                else run / "subsets" / resolved_subset / "methods" / method / "RETRIEVAL_COMPLETED.json"
            )
            freeze_json(marker_dest, method_marker_payload)
            per_method_records[method] = dict(
                path=str(marker_dest.resolve()),
                sha256=sha256_file(marker_dest),
                conditions=len(selected_specs),
                rankings=m_rankings,
                score_comparisons=m_comparisons,
            )
    finally:
        raw_encoder.close()

    if not is_full_panel:
        freeze_json(
            run / "subsets" / resolved_subset / "RETRIEVAL_COMPLETED.json",
            dict(
                status="completed",
                is_subset=True,
                subset_name=resolved_subset,
                selected_labels=selected_labels,
                plan_sha256=plan_sha,
                conditions=len(selected_specs),
                rankings=sum(v["rankings"] for v in per_method_records.values()),
                score_comparisons=sum(v["score_comparisons"] for v in per_method_records.values()),
                execution_device=device,
                model_sha256=model_sha,
                methods=per_method_records,
            ),
        )

    status(
        run,
        "retrieved" if is_full_panel else "retrieved_subset",
        is_subset=not is_full_panel,
        subset_name=resolved_subset,
        conditions=len(selected_specs),
        total_tasks=total_tasks,
    )
    return dict(
        status="completed",
        is_subset=not is_full_panel,
        subset_name=resolved_subset,
        selected_labels=selected_labels,
        plan_sha256=plan_sha,
    )


def verify(
    run: Path,
    device: str = "cuda",
    encoder_factory=None,
    selected_conditions: str | list[str] | tuple[str, ...] | None = None,
    subset_name: str | None = None,
) -> dict:
    """在独立进程与全新编码器下，重编码每张图并逐一核对选定条件（默认全量）全部原始分数与名次。"""
    run = assert_writable_run_dir(run)
    plan, plan_sha = validate_plan(run)
    selected_specs, is_full_panel = resolve_selected_conditions(plan, selected_conditions)
    resolved_subset = resolve_subset_name(selected_specs, is_full_panel, subset_name)
    selected_labels = [s["label"] for s in selected_specs]

    index_doc = _load_and_verify_defended_index(
        run,
        plan,
        plan_sha,
        selected_specs=selected_specs,
        require_sidecar=(encoder_factory is None),
    )
    bank_records = _load_bank_records(plan)

    for method in METHODS:
        rc = (
            run / "methods" / method / "RETRIEVAL_COMPLETED.json"
            if is_full_panel
            else run / "subsets" / resolved_subset / "methods" / method / "RETRIEVAL_COMPLETED.json"
        )
        if not rc.is_file():
            raise FileNotFoundError(f"缺少检索完成标记：{rc}")
        rc_doc = read_json(rc)
        if (
            rc_doc.get("status") != "completed"
            or rc_doc.get("plan_sha256") != plan_sha
            or rc_doc.get("conditions") != len(selected_specs)
            or (not is_full_panel and (rc_doc.get("subset_name") != resolved_subset or rc_doc.get("selected_labels") != selected_labels))
        ):
            raise ValueError(f"检索完成标记异常：{rc}")
        for lbl, rec in rc_doc.get("files", {}).items():
            check_file(rec["json"], rec["json_sha256"])
            check_file(rec["scores"], rec["scores_sha256"])

    raw_encoder, encoder = fresh_encoder(plan, device=device, encoder_factory=encoder_factory)
    model_sha = digest(raw_encoder.snapshot)
    total_comparisons = 0
    total_ranks = 0
    global_max_error = 0.0
    per_method_certs: dict[str, dict] = {}
    all_bound_files: dict[str, str] = {
        str((run / SCHEDULE_NAME).resolve()): plan_sha,
    }
    if is_full_panel:
        all_bound_files[str((run / "defended/INDEX.json").resolve())] = sha256_file(run / "defended/INDEX.json")
    else:
        for spec in selected_specs:
            cm = (run / "defended" / f"COMPLETED_{spec['label']}.json").resolve()
            all_bound_files[str(cm)] = sha256_file(cm)

    out_root = run if is_full_panel else (run / "subsets" / resolved_subset)
    expected_m_ranks = len(selected_specs) * plan["num_groups"] * len(BANK_NAMES) * plan["queries_per_group"]
    expected_m_cmp = expected_m_ranks * plan["pool_count"]
    expected_total_ranks = len(METHODS) * expected_m_ranks
    expected_total_cmp = len(METHODS) * expected_m_cmp

    try:
        query_vecs: dict[tuple[str, str], np.ndarray] = {}
        for method in METHODS:
            for group in plan["methods"][method]["groups"]:
                for q in group["queries"]:
                    qkey = (q["instruction"], q["query_image_sha256"])
                    if qkey not in query_vecs:
                        query_vecs[qkey] = encoder.encode_joint(
                            q["instruction"],
                            Path(q["query_image"]).resolve(),
                            image_weight=0.5,
                            _verified_sha256=q["query_image_sha256"],
                        )

        total_tasks = len(METHODS) * len(selected_specs)
        task_num = 0
        for method in METHODS:
            groups = plan["methods"][method]["groups"]
            m_comparisons = 0
            m_ranks = 0
            m_max_err = 0.0
            m_bindings: dict[str, str] = {}
            for spec in selected_specs:
                task_num += 1
                label = spec["label"]
                json_path, npz_path = _method_retrieval_paths(run, method, label)
                cond_items = index_doc["conditions"][label]
                saved_doc, saved_scores = _check_retrieval_artifact(
                    json_path, npz_path, plan, plan_sha, method, spec, cond_items, bank_records
                )
                if saved_doc["protocol"]["model_sha256"] != model_sha:
                    raise ValueError(f"独立复算编码器模型或权重摘要不符：{json_path}")

                status(
                    run,
                    "running",
                    stage="verify",
                    is_subset=not is_full_panel,
                    subset_name=resolved_subset,
                    method=method,
                    condition=label,
                    progress=f"{task_num}/{total_tasks}",
                )

                for g_idx, group in enumerate(groups):
                    gkey = group["key"]
                    g_saved = saved_doc["groups"][g_idx]
                    for b_idx, bank_name in enumerate(BANK_NAMES):
                        brec = bank_records[(method, gkey, bank_name)]
                        records = brec["records"]
                        b_saved = g_saved["banks"][bank_name]
                        src_idx = next(r["index"] for r in records if r["Id"] == group["source_memory_id"])
                        mem_vecs = [
                            encoder.encode_joint(
                                r["text"],
                                Path(cond_items[r["orig_sha256"]]["defended_path"]).resolve(),
                                image_weight=0.5,
                                _verified_sha256=cond_items[r["orig_sha256"]]["defended_sha256"],
                            )
                            for r in records
                        ]
                        for q_idx, (q, q_saved) in enumerate(zip(group["queries"], b_saved["rankings"])):
                            qv = query_vecs[(q["instruction"], q["query_image_sha256"])]
                            actual_scores = np.asarray([float(mv @ qv) for mv in mem_vecs], dtype=np.float32)
                            saved_row = saved_scores[g_idx, b_idx, q_idx, :]
                            diffs = np.abs(actual_scores.astype(np.float64) - saved_row.astype(np.float64))
                            if not np.isfinite(diffs).all():
                                raise ValueError(f"独立复算出现非有限差值：{method} {label} {gkey}")
                            row_max = float(diffs.max())
                            if row_max > 1e-6:
                                raise ValueError(
                                    f"独立复算分数超出容差 1e-6：{method} {label} {gkey} {bank_name} {q['task_id']} err={row_max}"
                                )
                            m_max_err = max(m_max_err, row_max)
                            order = _compute_ranking_order(actual_scores)
                            if order.index(src_idx) + 1 != q_saved["source_rank"]:
                                raise ValueError(f"独立复算源名次不符：{method} {label} {gkey} {q['task_id']}")
                            if [records[i]["Id"] for i in order[:3]] != q_saved["selected_ids"]:
                                raise ValueError(f"独立复算前三序列不符：{method} {label} {gkey}")
                            if [records[i]["Id"] for i in order[:5]] != q_saved["top5_ids"]:
                                raise ValueError(f"独立复算前五序列不符：{method} {label} {gkey}")
                            m_comparisons += len(records)
                            m_ranks += 1

                if label == "none":
                    _check_none_against_baseline(plan, method, saved_scores, saved_doc)

                j_sha = sha256_file(json_path)
                n_sha = sha256_file(npz_path)
                m_bindings[str(json_path.resolve())] = j_sha
                m_bindings[str(npz_path.resolve())] = n_sha
                all_bound_files[str(json_path.resolve())] = j_sha
                all_bound_files[str(npz_path.resolve())] = n_sha

            if m_ranks != expected_m_ranks or m_comparisons != expected_m_cmp:
                raise ValueError(f"{method} 复算数量不完整：ranks={m_ranks}, cmp={m_comparisons}")

            m_cert_path = out_root / "methods" / method / "INDEPENDENT_VERIFICATION.json"
            cert = dict(
                status="passed",
                is_subset=not is_full_panel,
                subset_name=resolved_subset,
                selected_labels=selected_labels,
                method=method,
                plan_sha256=plan_sha,
                fresh_process=(encoder_factory is None),
                fresh_encoder=True,
                execution_device=device,
                model_sha256=model_sha,
                no_original_vector_cache_read=True,
                conditions_verified=len(selected_specs),
                rankings_recomputed=m_ranks,
                score_comparisons=m_comparisons,
                max_score_error=m_max_err,
                files=m_bindings,
            )
            freeze_json(m_cert_path, cert)
            per_method_certs[method] = dict(
                path=str(m_cert_path.resolve()),
                sha256=sha256_file(m_cert_path),
                score_comparisons=m_comparisons,
                rankings_recomputed=m_ranks,
                max_score_error=m_max_err,
            )
            total_comparisons += m_comparisons
            total_ranks += m_ranks
            global_max_error = max(global_max_error, m_max_err)
            print(
                f"[verify] {method}: PASSED {m_ranks} rankings, {m_comparisons} scores, max_err={m_max_err:.3e}",
                flush=True,
            )
    finally:
        raw_encoder.close()

    if total_ranks != expected_total_ranks or total_comparisons != expected_total_cmp:
        raise ValueError("总独立复算规模未达预期覆盖")

    root_cert_path = out_root / "INDEPENDENT_VERIFICATION.json"
    root_cert = dict(
        status="passed",
        is_subset=not is_full_panel,
        subset_name=resolved_subset,
        selected_labels=selected_labels,
        plan_sha256=plan_sha,
        fresh_process=(encoder_factory is None),
        fresh_encoder=True,
        execution_device=device,
        model_sha256=model_sha,
        conditions_verified=len(selected_specs),
        rankings_recomputed=total_ranks,
        score_comparisons=total_comparisons,
        max_score_error=global_max_error,
        methods=per_method_certs,
        files=all_bound_files,
    )
    freeze_json(root_cert_path, root_cert)
    status(
        run,
        "verified" if is_full_panel else "verified_subset",
        is_subset=not is_full_panel,
        subset_name=resolved_subset,
        conditions_verified=len(selected_specs),
        rankings_recomputed=total_ranks,
        score_comparisons=total_comparisons,
        max_score_error=global_max_error,
    )
    return root_cert


def _aggregate_stats(ranks: list[int], pool_count: int) -> dict:
    if not ranks or any(type(x) is not int or not 1 <= x <= pool_count for x in ranks):
        raise ValueError("名次数据异常")
    n = len(ranks)
    top1 = sum(x == 1 for x in ranks)
    top3 = sum(x <= 3 for x in ranks)
    top5 = sum(x <= 5 for x in ranks)
    return dict(
        n=n,
        top1=top1,
        top3=top3,
        top5=top5,
        top1_rate=top1 / n,
        top3_rate=top3 / n,
        top5_rate=top5 / n,
        mean_rank=sum(ranks) / n,
    )


def _compute_paired_summary(
    docs_by_label: dict[str, dict],
    labels: list[str],
    group_keys: list[str] | None = None,
) -> dict:
    clean_rows = []
    mod_rows = []
    for lbl in labels:
        doc = docs_by_label[lbl]
        c_qs = []
        m_qs = []
        for g_doc in doc["groups"]:
            if group_keys is not None and g_doc["group"] not in group_keys:
                continue
            gkey = g_doc["group"]
            for q in g_doc["banks"]["clean"]["rankings"]:
                c_qs.append(dict(task_id=f"{gkey}__{q['task_id']}", rank=q["source_rank"]))
            for q in g_doc["banks"]["modified"]["rankings"]:
                m_qs.append(dict(task_id=f"{gkey}__{q['task_id']}", rank=q["source_rank"]))
        clean_rows.append(dict(condition=dict(label=lbl), queries=c_qs))
        mod_rows.append(dict(condition=dict(label=lbl), queries=m_qs))

    paired = paired_rank_metrics(clean_rows, mod_rows)
    clean_ranks = [q["rank"] for r in clean_rows for q in r["queries"]]
    mod_ranks = [q["rank"] for r in mod_rows for q in r["queries"]]
    retained_top3 = sum(a <= 3 and b <= 3 for a, b in zip(clean_ranks, mod_ranks))
    neither_top3 = sum(a > 3 and b > 3 for a, b in zip(clean_ranks, mod_ranks))
    top1_gain = (sum(b == 1 for b in mod_ranks) - sum(a == 1 for a in clean_ranks)) / len(clean_ranks)
    top5_gain = (sum(b <= 5 for b in mod_ranks) - sum(a <= 5 for a in clean_ranks)) / len(clean_ranks)
    return dict(
        n=paired["n"],
        new_top3=paired["new_top3"],
        lost_top3=paired["lost_top3"],
        retained_top3=retained_top3,
        neither_top3=neither_top3,
        top3_gain=paired["top3_gain"],
        top3_gain_pp=paired["top3_gain"] * 100.0,
        top1_gain=top1_gain,
        top1_gain_pp=top1_gain * 100.0,
        top5_gain=top5_gain,
        top5_gain_pp=top5_gain * 100.0,
        mean_rank_gain=paired["mean_rank_gain"],
    )


def _freeze_text(path: Path, content: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise ValueError(f"拒绝覆盖已有不同报告文件：{path}")
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def _fmt_frac(stats: dict, k: int) -> str:
    return f"{stats[f'top{k}']}/{stats['n']} ({stats[f'top{k}_rate']:.2%})"


def report(run: Path) -> dict:
    """校验全套独立复核证据后，生成 SUMMARY.json、ALL_RANKINGS.csv、REPORT.md 与 COMPLETED.json。"""
    run = assert_writable_run_dir(run)
    plan, plan_sha = validate_plan(run)

    root_cert_path = run / "INDEPENDENT_VERIFICATION.json"
    if not root_cert_path.is_file():
        raise ValueError("缺少根级独立复核证书 INDEPENDENT_VERIFICATION.json，禁止生成报告")
    root_cert = read_json(root_cert_path)
    if (
        root_cert.get("status") != "passed"
        or root_cert.get("plan_sha256") != plan_sha
        or root_cert.get("rankings_recomputed") != plan["expected_rankings"]
        or root_cert.get("score_comparisons") != plan["expected_score_comparisons"]
        or not math.isfinite(root_cert.get("max_score_error", math.inf))
        or root_cert["max_score_error"] > 1e-6
    ):
        raise ValueError("独立复核证书状态、覆盖范围或误差校验未通过")

    for file_str, expected_sha in root_cert.get("files", {}).items():
        check_file(file_str, expected_sha)

    docs: dict[str, dict[str, dict]] = {}
    for method in METHODS:
        m_cert_path = run / "methods" / method / "INDEPENDENT_VERIFICATION.json"
        if not m_cert_path.is_file():
            raise ValueError(f"缺少方法独立复核证书：{m_cert_path}")
        m_cert = read_json(m_cert_path)
        if m_cert.get("status") != "passed" or m_cert.get("plan_sha256") != plan_sha:
            raise ValueError(f"方法独立复核证书无效：{m_cert_path}")
        for file_str, expected_sha in m_cert.get("files", {}).items():
            check_file(file_str, expected_sha)
        docs[method] = {}
        for spec in plan["conditions"]:
            lbl = spec["label"]
            jpath, _ = _method_retrieval_paths(run, method, lbl)
            doc = read_json(jpath)
            if doc["protocol"]["model_sha256"] != root_cert["model_sha256"]:
                raise ValueError(f"结果编码器身份与独立复核证书不一致：{jpath}")
            docs[method][lbl] = doc

    pool_count = int(plan["pool_count"])
    all_group_keys = [g["key"] for g in plan["methods"][METHODS[0]]["groups"]]
    group_names = {g["key"]: g["name"] for g in plan["methods"][METHODS[0]]["groups"]}

    table_rows_summary = []
    for row_spec in plan["table_groups"]:
        title = row_spec["title"]
        labels = row_spec["labels"]
        methods_row = {}
        for method in METHODS:
            clean_ranks = [
                q["source_rank"]
                for lbl in labels
                for g in docs[method][lbl]["groups"]
                for q in g["banks"]["clean"]["rankings"]
            ]
            mod_ranks = [
                q["source_rank"]
                for lbl in labels
                for g in docs[method][lbl]["groups"]
                for q in g["banks"]["modified"]["rankings"]
            ]
            per_group = {}
            for gkey in all_group_keys:
                gc = [
                    q["source_rank"]
                    for lbl in labels
                    for g in docs[method][lbl]["groups"]
                    if g["group"] == gkey
                    for q in g["banks"]["clean"]["rankings"]
                ]
                gm = [
                    q["source_rank"]
                    for lbl in labels
                    for g in docs[method][lbl]["groups"]
                    if g["group"] == gkey
                    for q in g["banks"]["modified"]["rankings"]
                ]
                per_group[gkey] = dict(
                    name=group_names[gkey],
                    clean=_aggregate_stats(gc, pool_count),
                    modified=_aggregate_stats(gm, pool_count),
                    paired=_compute_paired_summary(docs[method], labels, group_keys=[gkey]),
                )
            methods_row[method] = dict(
                clean=_aggregate_stats(clean_ranks, pool_count),
                modified=_aggregate_stats(mod_ranks, pool_count),
                paired=_compute_paired_summary(docs[method], labels),
                per_group=per_group,
            )
        first_spec = next(s for s in plan["conditions"] if s["label"] == labels[0])
        table_rows_summary.append(
            dict(
                title=title,
                labels=labels,
                kind=first_spec["kind"],
                seed_count=len(labels),
                methods=methods_row,
            )
        )

    conditions_26_summary = []
    for spec in plan["conditions"]:
        lbl = spec["label"]
        per_m = {}
        for method in METHODS:
            c_ranks = [
                q["source_rank"]
                for g in docs[method][lbl]["groups"]
                for q in g["banks"]["clean"]["rankings"]
            ]
            m_ranks = [
                q["source_rank"]
                for g in docs[method][lbl]["groups"]
                for q in g["banks"]["modified"]["rankings"]
            ]
            per_m[method] = dict(
                clean=_aggregate_stats(c_ranks, pool_count),
                modified=_aggregate_stats(m_ranks, pool_count),
                paired=_compute_paired_summary(docs[method], [lbl]),
            )
        conditions_26_summary.append(dict(condition=spec, methods=per_m))

    baseline_meta = {}
    for method in METHODS:
        none_doc = docs[method]["none"]
        if "baseline_check_vs_20261110" in none_doc:
            baseline_meta[method] = none_doc["baseline_check_vs_20261110"]

    summary = dict(
        schema_version=1,
        status="completed",
        environment="env2_feature_text_plus_vague_photo",
        scope="retrieval_only_no_shopping_no_external_model_calls",
        plan_sha256=plan_sha,
        pool_count=pool_count,
        conditions_count=len(plan["conditions"]),
        table_rows_count=len(plan["table_groups"]),
        rankings=plan["expected_rankings"],
        score_comparisons=plan["expected_score_comparisons"],
        shopping_calls=0,
        generation_calls=0,
        external_model_calls=0,
        table_rows=table_rows_summary,
        conditions_26=conditions_26_summary,
        baseline_none_comparison_with_20261110=baseline_meta,
        independent_verification=dict(
            path=str(root_cert_path.resolve()),
            sha256=sha256_file(root_cert_path),
            rankings_recomputed=root_cert["rankings_recomputed"],
            score_comparisons=root_cert["score_comparisons"],
            max_score_error=root_cert["max_score_error"],
        ),
        caveats=plan["caveats"],
    )

    label_to_row_title = {lbl: row["title"] for row in plan["table_groups"] for lbl in row["labels"]}
    csv_path = run / "ALL_RANKINGS.csv"
    tmp_csv = csv_path.with_name(csv_path.name + ".tmp")
    with tmp_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "method",
                "method_name",
                "condition",
                "table_group",
                "condition_kind",
                "timestep",
                "seed",
                "group",
                "group_name",
                "bank",
                "task_id",
                "source_memory_id",
                "source_rank",
                "source_score",
                "top1_hit",
                "top3_hit",
                "top5_hit",
                "top1_id",
                "top3_ids",
            ]
        )
        for method in METHODS:
            for spec in plan["conditions"]:
                lbl = spec["label"]
                doc = docs[method][lbl]
                for g in doc["groups"]:
                    for bank_name in BANK_NAMES:
                        for q in g["banks"][bank_name]["rankings"]:
                            r = q["source_rank"]
                            writer.writerow(
                                [
                                    method,
                                    METHOD_NAMES[method],
                                    lbl,
                                    label_to_row_title[lbl],
                                    spec["kind"],
                                    spec.get("timestep", ""),
                                    spec.get("seed", ""),
                                    g["group"],
                                    g["name"],
                                    bank_name,
                                    q["task_id"],
                                    g["source_memory_id"],
                                    r,
                                    f"{q['source_score']:.8f}",
                                    int(r == 1),
                                    int(r <= 3),
                                    int(r <= 5),
                                    q["selected_ids"][0] if q["selected_ids"] else "",
                                    "|".join(q["selected_ids"]),
                                ]
                            )
    if csv_path.exists():
        if csv_path.read_text(encoding="utf-8") != tmp_csv.read_text(encoding="utf-8"):
            tmp_csv.unlink()
            raise ValueError(f"拒绝覆盖已存在且内容不同的 CSV 文件：{csv_path}")
        tmp_csv.unlink()
    else:
        tmp_csv.replace(csv_path)

    none_row = table_rows_summary[0]
    lucid_none_cln = none_row["methods"]["lucid"]["clean"]
    lucid_none_mod = none_row["methods"]["lucid"]["modified"]
    per_none_mod = none_row["methods"]["perlage_no_pixel"]["modified"]

    lines = [
        "# 环境二（特征文字 + 独立模糊照片）防御条件下的共享记忆检索评估报告",
        "",
        "## 一、实验协议与范围说明",
        "",
        f"- **实验状态**：环境二（特征文字 + 独立生成无品牌模糊照片），仅评估检索排序；严禁购物、严禁外部策略大模型、不重新训练、不重新选图。",
        f"- **查询与记忆键**：查询采用 20 条固定英文特点购买意向 + 每组 1 张独立模糊照片（图文权重各 0.5）；**所有查询图片保持原样不加防御**。记忆库共 {pool_count} 条记录（180 条正常背景 + 9 条对应组原商品模拟参考），检索键为商品名称 + 已确认品牌 + 特点属性 + 商品图片（图文权重各 0.5），**记忆库中全部图片施加完全相同的防御**。",
        f"- **无防御基线核验**：与原冻结无防御基线（`groups9_multimodal_20261110`）完全一致——原图库前三 `{lucid_none_cln['top3']}/{lucid_none_cln['n']} ({lucid_none_cln['top3_rate']:.2%})`；原始像素方法（`lucid`）修改库前三 `{lucid_none_mod['top3']}/{lucid_none_mod['n']} ({lucid_none_mod['top3_rate']:.2%})`；无像素感知方法（`perlage_no_pixel`）修改库前三 `{per_none_mod['top3']}/{per_none_mod['n']} ({per_none_mod['top3_rate']:.2%})`。",
        f"- **独立复算证明**：全新进程 + 全新加载公开 CLIP（对比语言-图像预训练模型）`ViT-B-32` (`QuickGELU`) 编码器独立完成全部 **{root_cert['rankings_recomputed']:,}** 次全库排名、**{root_cert['score_comparisons']:,}** 个候选分数逐项核验，最大分数误差 `{root_cert['max_score_error']:.3e}`（容差 `1e-6`）。",
        "",
        "## 二、两方法十四行防御条件横向对比（环境二总表）",
        "",
        "说明：`前三增量` = 修改库前三命中率 − 原图库前三命中率（单位：百分点 `pp`）；扩散行 `n=900`（9 组 × 20 请求 × 5 个留出随机种子 `1001..1005`），其余单条件行 `n=180`（9 组 × 20 请求）。同一请求的不同扩散种子并非独立需求。",
        "",
        "| 防御条件 | 原图前三（共同基准） | Lucid 修改前三 | Lucid 前三增量(pp) | Lucid 修改前五 | Lucid 修改平均名次 | PerLAGE(无像素) 修改前三 | PerLAGE 前三增量(pp) | PerLAGE 修改前五 | PerLAGE 修改平均名次 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table_rows_summary:
        lc = row["methods"]["lucid"]["clean"]
        lm = row["methods"]["lucid"]["modified"]
        lp = row["methods"]["lucid"]["paired"]
        pm = row["methods"]["perlage_no_pixel"]["modified"]
        pp = row["methods"]["perlage_no_pixel"]["paired"]
        lines.append(
            "| "
            + " | ".join(
                [
                    row["title"],
                    _fmt_frac(lc, 3),
                    _fmt_frac(lm, 3),
                    f"{lp['top3_gain_pp']:+.2f}",
                    _fmt_frac(lm, 5),
                    f"{lm['mean_rank']:.2f}",
                    _fmt_frac(pm, 3),
                    f"{pp['top3_gain_pp']:+.2f}",
                    _fmt_frac(pm, 5),
                    f"{pm['mean_rank']:.2f}",
                ]
            )
            + " |"
        )

    for method in METHODS:
        lines += [
            "",
            f"## 三、{METHOD_NAMES[method]}（`{method}`）十四行详细指标",
            "",
            "| 防御条件 | 原图第一 | 修改第一 | 原图前三 | 修改前三 | 原图前五 | 修改前五 | 前三增量(pp) | 原图平均名次 | 修改平均名次 | 平均名次改善 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in table_rows_summary:
            c = row["methods"][method]["clean"]
            m = row["methods"][method]["modified"]
            p = row["methods"][method]["paired"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        row["title"],
                        _fmt_frac(c, 1),
                        _fmt_frac(m, 1),
                        _fmt_frac(c, 3),
                        _fmt_frac(m, 3),
                        _fmt_frac(c, 5),
                        _fmt_frac(m, 5),
                        f"{p['top3_gain_pp']:+.2f}",
                        f"{c['mean_rank']:.2f}",
                        f"{m['mean_rank']:.2f}",
                        f"{p['mean_rank_gain']:+.2f}",
                    ]
                )
                + " |"
            )

    lines += [
        "",
        "## 四、全部二十六项防御设置逐种子明细（完整保留扩散种子 `1001..1005`）",
        "",
        "| 方法 | 防御设置 | 种类 | 原图第一 | 原图前三 | 原图前五 | 原图均名次 | 修改第一 | 修改前三 | 修改前五 | 修改均名次 | 前三增量(pp) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in conditions_26_summary:
        spec = item["condition"]
        for method in METHODS:
            c = item["methods"][method]["clean"]
            m = item["methods"][method]["modified"]
            p = item["methods"][method]["paired"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        METHOD_NAMES[method],
                        spec["label"],
                        spec["kind"],
                        _fmt_frac(c, 1),
                        _fmt_frac(c, 3),
                        _fmt_frac(c, 5),
                        f"{c['mean_rank']:.2f}",
                        _fmt_frac(m, 1),
                        _fmt_frac(m, 3),
                        _fmt_frac(m, 5),
                        f"{m['mean_rank']:.2f}",
                        f"{p['top3_gain_pp']:+.2f}",
                    ]
                )
                + " |"
            )

    for method in METHODS:
        lines += [
            "",
            f"## 五、{METHOD_NAMES[method]}（`{method}`）九组修改库前三命中矩阵（Clean 原图库基准括号内）",
            "",
            "| 防御条件 | 总体 修改(原图) | " + " | ".join(group_names[k] for k in all_group_keys) + " |",
            "|---|---:|" + "|".join(["---:"] * len(all_group_keys)) + "|",
        ]
        for row in table_rows_summary:
            m_all = row["methods"][method]["modified"]
            c_all = row["methods"][method]["clean"]
            cells = [f"{m_all['top3']}/{m_all['n']} ({c_all['top3']})"]
            for gk in all_group_keys:
                gm = row["methods"][method]["per_group"][gk]["modified"]
                gc = row["methods"][method]["per_group"][gk]["clean"]
                cells.append(f"{gm['top3']}/{gm['n']} ({gc['top3']})")
            lines.append("| " + row["title"] + " | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## 六、方法局限与解释边界（必读披露）",
        "",
        *[f"- {c}" for c in plan["caveats"]],
        f"- 原始分数矩阵、逐条件/逐组 JSON 与 18,720 条完整排名 CSV（逗号分隔值文件）均保存在同目录，且受 `INDEPENDENT_VERIFICATION.json` SHA256 绑定。",
        "",
    ]

    freeze_json(run / "SUMMARY.json", summary)
    report_path = run / "REPORT.md"
    _freeze_text(report_path, "\n".join(lines))

    completion_file = run / "COMPLETED.json"
    completed_at = read_json(completion_file)["completed_at"] if completion_file.is_file() else utcnow()
    freeze_json(
        completion_file,
        dict(
            status="completed",
            completed_at=completed_at,
            environment="env2_feature_text_plus_vague_photo",
            plan_sha256=plan_sha,
            rankings=plan["expected_rankings"],
            score_comparisons=plan["expected_score_comparisons"],
            methods=list(METHODS),
            shopping_calls=0,
            generation_calls=0,
            external_model_calls=0,
            summary_sha256=sha256_file(run / "SUMMARY.json"),
            report_sha256=sha256_file(report_path),
            csv_sha256=sha256_file(csv_path),
            independent_verification_sha256=sha256_file(root_cert_path),
        ),
    )
    status(
        run,
        "completed",
        rankings=plan["expected_rankings"],
        score_comparisons=plan["expected_score_comparisons"],
        report_path=str(report_path.resolve()),
    )
    return summary


def _stage_done(run: Path, stage: str, plan_sha: str) -> bool:
    if stage == "prepare":
        p = run / "PREPARATION_COMPLETED.json"
        return p.is_file() and read_json(p).get("plan_sha256") == plan_sha
    if stage == "defend":
        p = run / "DEFENSE_COMPLETED.json"
        return p.is_file() and read_json(p).get("status") == "completed" and read_json(p).get("plan_sha256") == plan_sha
    if stage == "retrieve":
        for m in METHODS:
            p = run / "methods" / m / "RETRIEVAL_COMPLETED.json"
            if not p.is_file():
                return False
            d = read_json(p)
            if d.get("status") != "completed" or d.get("plan_sha256") != plan_sha:
                return False
        return True
    if stage == "verify":
        p = run / "INDEPENDENT_VERIFICATION.json"
        return p.is_file() and read_json(p).get("status") == "passed" and read_json(p).get("plan_sha256") == plan_sha
    if stage == "report":
        p = run / "COMPLETED.json"
        return p.is_file() and read_json(p).get("status") == "completed" and read_json(p).get("plan_sha256") == plan_sha
    return False


def queue(
    run: Path = DEFAULT_RUN,
    *,
    device: str = "cuda",
    origin_plan: Path = DEFAULT_ORIGIN_PLAN,
    origin_retrieval_dir: Path | None = DEFAULT_ORIGIN_RETRIEVAL_DIR,
    expected_origin_sha256: str | None = EXPECTED_ORIGIN_PLAN_SHA256,
) -> dict:
    """分阶段独立子进程顺序编排：prepare -> defend (SD1.5) -> retrieve (CLIP) -> verify (fresh CLIP) -> report。"""
    run = assert_writable_run_dir(run)
    handle = lock(run / "RUN.lock")
    try:
        if not (run / SCHEDULE_NAME).is_file():
            prepare(
                run,
                origin_plan=origin_plan,
                origin_retrieval_dir=origin_retrieval_dir,
                expected_origin_sha256=expected_origin_sha256,
            )
        plan, plan_sha = validate_plan(run)

        for stage in ("defend", "retrieve", "verify"):
            if _stage_done(run, stage, plan_sha):
                continue
            status(run, "running", parent_stage=stage)
            log_path = run / "logs" / f"{stage}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                "-X",
                "utf8",
                "-B",
                "-u",
                str(Path(__file__).resolve()),
                "--run",
                str(run),
                "--stage",
                stage,
                "--device",
                device,
            ]
            with log_path.open("a", encoding="utf-8") as log_f:
                child = subprocess.Popen(cmd, cwd=PROJECT, stdout=log_f, stderr=subprocess.STDOUT)
                save_json(
                    run / "ACTIVE_CHILD.json",
                    dict(pid=child.pid, stage=stage, command=cmd, launched_at=utcnow()),
                )
                code = child.wait()
            if code != 0 or not _stage_done(run, stage, plan_sha):
                raise RuntimeError(f"子阶段执行失败：stage={stage} exit_code={code}")

        summary = report(run)
        return summary
    finally:
        handle.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="环境二（特征文字+模糊照片）26条件图像防御检索评估")
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument(
        "--stage",
        choices=["prepare", "defend", "retrieve", "verify", "report", "all"],
        default="all",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--origin-plan", type=Path, default=DEFAULT_ORIGIN_PLAN)
    args = parser.parse_args()

    run = assert_writable_run_dir(args.run)
    try:
        origin_resolved = args.origin_plan.resolve()
        expected_sha = EXPECTED_ORIGIN_PLAN_SHA256 if origin_resolved == DEFAULT_ORIGIN_PLAN.resolve() else None
        origin_rdir = DEFAULT_ORIGIN_RETRIEVAL_DIR if expected_sha else origin_resolved.parent

        if args.stage == "prepare":
            prepare(
                run,
                origin_plan=origin_resolved,
                origin_retrieval_dir=origin_rdir,
                expected_origin_sha256=expected_sha,
            )
        elif args.stage == "defend":
            defend(run, device=args.device)
        elif args.stage == "retrieve":
            retrieve(run, device=args.device)
        elif args.stage == "verify":
            verify(run, device=args.device)
        elif args.stage == "report":
            report(run)
        elif args.stage == "all":
            queue(
                run,
                device=args.device,
                origin_plan=origin_resolved,
                origin_retrieval_dir=origin_rdir,
                expected_origin_sha256=expected_sha,
            )
    except Exception as exc:
        detail = dict(
            status="failed",
            failed_at=utcnow(),
            pid=os.getpid(),
            stage=args.stage,
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        save_json(
            run / "failures" / (time.strftime("%Y%m%dT%H%M%S", time.gmtime()) + f"_{os.getpid()}.json"),
            detail,
        )
        status(run, "failed", stage=args.stage, error_type=type(exc).__name__, error=str(exc))
        raise


if __name__ == "__main__":
    main()
