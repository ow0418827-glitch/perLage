"""重选洗发水组 (07_shampoo: B09B63YYCW -> B00BDJODWI) Lucid (8/255, 1000步) 独立生成与检索评测流水线：
在完全独立实验目录中运行：
  prepare -> generate_lucid (epsilon=8/255, steps=1000, text_weight=1.0, 三代理联合优化)
  -> freeze_images -> evaluate (joint + text 双查询模式、189共享池单条目替换、clean/lucid 双库)
  -> verify (独立子进程从零重编码复算 15,120 个候选分数)
  -> report (汇总指标、视觉检验拼图、与 ambient_coop 及旧洗发水组 B08XYBS678 只读对照)。

严格遵守：
1. 训练与生成期严禁读取共享记忆池、proposed_pool_error_entry.json、正式查询清单及任何既有实验输出；
2. 完整生成 1000 步 Lucid 成图并冻结后，方可在隔离银行副本中替换洗发水单条目并评测；
3. 仅评估检索排名 (shopping_enabled=False, model_api_calls_enabled=False)，不修改任何已有数据集或实验结果。
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

os.environ.update(
    HF_HUB_OFFLINE="1",
    TRANSFORMERS_OFFLINE="1",
    TOKENIZERS_PARALLELISM="false",
    PYTHONDONTWRITEBYTECODE="1",
    PYTHONUTF8="1",
)

import numpy as np
from PIL import Image, ImageDraw
import torch

from src.visual.audit import digest, freeze_json, read_json, save_json
from src.visual.image_utils import sha256_file
from src.visual.io import decoded_image_audit
from src.visual._perlage.components.colour import read_rgb, print_region_mask
from src.visual._perlage.components.core import source_ssim

BASE = PROJECT / "output/experiment_base_groups9_20261109"
RESELECT_DIR = PROJECT / "output/public_pair_selection/shampoo_reselect_20261112"
RESELECT_COMP = RESELECT_DIR / "comparison_package"
RESELECT_PKG = RESELECT_COMP / "07_shampoo"
QUERY_MULTIMODAL_RUN = PROJECT / "output/groups9_multimodal_20261110"

# 只读对照路径（仅在 report 阶段读取汇总，训练与检索绝不访问）
HIST_AMBIENT_RESELECT_RUN = PROJECT / "output/perlage_ambient_coop_shampoo_reselect_20261112"
HIST_AMBIENT_RUN = PROJECT / "output/perlage_ambient_coop_groups79_20261112"
HIST_NO_PIXEL_RUN = PROJECT / "output/perlage_no_pixel_groups9_20261109"
HIST_LUCID_RUN = PROJECT / "output/lucid_groups9_20261109"
HIST_SHAMPOO_MODES = PROJECT / "output/analysis_shampoo_query_modes_20261112"
DEFAULT_RUN = PROJECT / "output/lucid_shampoo_reselect_20261112"

GROUP_KEY = "07_shampoo"
METHOD = "lucid"
REQUESTED_STEPS = 1000
EPSILON_LEVELS = 8
TEXT_WEIGHT = 1.0
OLD_SHAMPOO_ASIN = "B08XYBS678"
OLD_SHAMPOO_MEMORY_ID = "groups9_reference_07_shampoo_B08XYBS678"
NEW_SHAMPOO_ASIN = "B09B63YYCW"
NEW_SHAMPOO_MEMORY_ID = "groups9_reference_07_shampoo_B09B63YYCW"
TARGET_ASIN = "B00BDJODWI"
BANKS = ("clean", "lucid")
QUERY_MODES = ("joint", "text")


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _check(path: str | Path, expected: str) -> None:
    path_obj = Path(path)
    if not expected or not path_obj.is_file() or sha256_file(path_obj) != expected:
        raise ValueError(f"冻结文件缺失或内容摘要改变：{path}")


def lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    if handle.tell() == 0:
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
        raise RuntimeError("已有本轮队列或工作进程正在运行，拒绝重复并发启动") from None
    return handle


def status(run: Path, state: str, **extra) -> None:
    value = dict(status=state, updated_at=now(), pid=os.getpid(), **extra)
    save_json(run / "RUN_STATUS.json", value)
    labels = {
        "running": "运行中",
        "completed": "全部完成",
        "failed": "技术失败后停止",
        "paused": "已保存断点暂停",
    }
    lines = [
        "# 重选洗发水组（B09B63YYCW -> B00BDJODWI）Lucid (8/255, 1000步) 实验运行状态",
        "",
        f"- 当前状态：{labels.get(state, state)}",
        f"- 更新时间：{value['updated_at']}",
        f"- 进程 PID：{os.getpid()}",
    ]
    for key in ("stage", "mode", "bank", "log", "report", "message"):
        if key in extra and extra[key] is not None:
            lines.append(f"- {key}：`{extra[key]}`")
    lines += [
        "",
        "生成日志见 `logs/generate_lucid.log`；",
        "完成后将自动生成 `REPORT.md`、`SUMMARY.json` 与 `COMPLETED.json`。",
        "",
    ]
    (run / "RUN_STATUS.md").write_text("\n".join(lines), encoding="utf-8")


def output_root(path: Path | str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT / p
    run = p.resolve()
    forbidden = {
        BASE.resolve(),
        RESELECT_DIR.resolve(),
        QUERY_MULTIMODAL_RUN.resolve(),
        HIST_AMBIENT_RESELECT_RUN.resolve(),
        HIST_AMBIENT_RUN.resolve(),
        HIST_NO_PIXEL_RUN.resolve(),
        HIST_LUCID_RUN.resolve(),
        HIST_SHAMPOO_MODES.resolve(),
    }
    if run.parent != (PROJECT / "output").resolve() or run in forbidden:
        raise ValueError(f"只允许写入项目 output 下的全新独立实验目录，拒绝写入只读目录：{run}")
    return run


def implementation_files() -> dict[str, str]:
    paths = [Path(__file__)]
    lucid_dir = PROJECT / "src/visual/_lucid"
    if lucid_dir.exists():
        paths += sorted(lucid_dir.glob("*.py"))
    paths += [
        PROJECT / p
        for p in (
            "src/visual/optimization.py",
            "src/visual/config/optimization.json",
            "src/visual/title_alignment.py",
            "src/visual/attack_completion.py",
            "src/visual/catalog_training_guard.py",
            "src/visual/io.py",
            "src/visual/visual_retrieval.py",
            "src/visual/encoders/joint.py",
            "src/visual/encoders/clip.py",
            "src/visual/encoders/backbone.py",
            "src/experiments/query_contract.py",
            "src/experiments/shared_evaluation.py",
            "src/agent/memory/shared_pool.py",
        )
    ]
    unique = {str(p.resolve()): sha256_file(p) for p in paths if p.is_file()}
    return dict(sorted(unique.items()))


def _private_paths_for_training(run: Path) -> list[Path]:
    return [
        BASE / "shared_memory",
        BASE / "evaluation_inputs",
        BASE / "groups",
        QUERY_MULTIMODAL_RUN,
        HIST_AMBIENT_RESELECT_RUN,
        HIST_AMBIENT_RUN,
        HIST_NO_PIXEL_RUN,
        HIST_LUCID_RUN,
        HIST_SHAMPOO_MODES,
        RESELECT_COMP / "proposed_pool_error_entry.json",
        run / "retrieval",
        run / "banks",
        run / "EVALUATION_INPUTS.json",
    ]


# ==============================================================================
# Stage 1: prepare (严格只读公开选图资产与元数据)
# ==============================================================================


def prepare(run: Path) -> dict:
    """准备阶段：仅读取公开选图资料与公开元数据；严禁打开共享库、错误模板、正式查询或历史评测结果。"""
    from src.visual._lucid.config import resolve_optimization

    run = output_root(run)
    run.mkdir(parents=True, exist_ok=True)

    lucid_cfg_path = RESELECT_PKG / "lucid_config.json"
    definition_path = RESELECT_PKG / "definition.json"
    features_path = RESELECT_PKG / "features.json"
    metadata_path = RESELECT_COMP / "public_metadata/source_and_target_public_metadata.json"
    opt_cfg_path = PROJECT / "src/visual/config/optimization.json"
    source_path = (RESELECT_PKG / "source.jpg").resolve()
    target_path = (RESELECT_PKG / "target.jpg").resolve()

    lucid_cfg = read_json(lucid_cfg_path)
    public_def = read_json(definition_path)
    features_doc = read_json(features_path)
    catalog_products = read_json(metadata_path)["products"]

    if (
        lucid_cfg.get("method") != METHOD
        or lucid_cfg.get("steps") != REQUESTED_STEPS
        or lucid_cfg.get("epsilon_levels") != EPSILON_LEVELS
        or float(lucid_cfg.get("text_weight", -1.0)) != TEXT_WEIGHT
    ):
        raise ValueError(f"lucid_config.json 参数与约定 (8/255, 1000步, text_weight=1.0) 不一致：{lucid_cfg}")

    source_id = public_def["source"]["id"]
    target_id = public_def["target"]["id"]
    if source_id != NEW_SHAMPOO_ASIN or target_id != TARGET_ASIN:
        raise ValueError(f"重选图对标识不符：期望 {NEW_SHAMPOO_ASIN}->{TARGET_ASIN}，实际 {source_id}->{target_id}")

    source_feature = features_doc["source"]
    target_feature = features_doc["target"]
    source_catalog = catalog_products[source_id]

    if source_feature.get("asin") != source_id or target_feature.get("asin") != target_id:
        raise ValueError("features.json 与 definition.json 公开商品编号不一致")
    if (
        source_feature["title"] != public_def["data"]["source_title"]
        or public_def["source"]["title"] != source_feature["title"]
        or source_catalog["Title"] != source_feature["title"]
    ):
        raise ValueError("公开源商品名不一致")

    target_text = lucid_cfg["target_text"]
    if (
        target_text != public_def["target_feature_text"]
        or target_text != target_feature["canonical_sentence"]
        or target_text != public_def["lucid_alignment_config"]["target_text"]
        or public_def["lucid_alignment_config"].get("steps") != REQUESTED_STEPS
        or float(public_def["lucid_alignment_config"].get("text_weight", -1.0)) != TEXT_WEIGHT
    ):
        raise ValueError("Lucid 训练目标特点正文在各公开配置间不一致")

    # 在 CPU 侧提前验证基础三代理优化配置合法
    resolve_optimization(opt_cfg_path).validate()

    source_public_record = dict(
        id=source_id,
        title=source_feature["title"],
        brand=source_feature.get("brand", ""),
        attributes=public_def["source"]["attributes"],
        feature_text=source_feature["canonical_sentence"],
        provenance=source_feature["source"],
        feature_evidence=source_feature["feature_evidence"],
    )

    bindings = {
        str(p.resolve()): sha256_file(p)
        for p in (
            lucid_cfg_path,
            definition_path,
            features_path,
            metadata_path,
            opt_cfg_path,
            source_path,
            target_path,
        )
    }

    plan = dict(
        schema_version=1,
        experiment="lucid_shampoo_reselect_20261112",
        group=GROUP_KEY,
        groups=[GROUP_KEY],
        method=METHOD,
        source_id=source_id,
        target_id=target_id,
        replaces_memory_id=OLD_SHAMPOO_MEMORY_ID,
        new_memory_id=NEW_SHAMPOO_MEMORY_ID,
        generation_options=dict(
            method=METHOD,
            steps=REQUESTED_STEPS,
            epsilon_levels=EPSILON_LEVELS,
            epsilon=EPSILON_LEVELS / 255.0,
            text_weight=TEXT_WEIGHT,
            target_text=target_text,
            text_alignment_source="target_product_features",
            seed=read_json(opt_cfg_path)["seed"],
            optimization_config=str(opt_cfg_path.resolve()),
        ),
        generation_inputs={
            GROUP_KEY: dict(
                source_image=str(source_path),
                target_image=str(target_path),
                lucid_config=lucid_cfg,
                public_definition=public_def,
                source_public_record=source_public_record,
                method_root=str((run / "generation" / METHOD / GROUP_KEY).resolve()),
                files=bindings,
            )
        },
        implementation=implementation_files(),
        query_modes=list(QUERY_MODES),
        query_weights={"joint": 0.5, "text": 0.0},
        key_image_weight=0.5,
        index_text="product_feature",
        shared_pool_count=189,
        pool_replacement_scope="single_group_07_shampoo_B08XYBS678_replaced_with_B09B63YYCW_in_isolated_copy",
        all_images_frozen_before_evaluation=True,
        pixel_optimization=True,
        shopping_enabled=False,
        model_api_calls_enabled=False,
        formal_rankings_used_for_parameter_selection=False,
        training_reads_shared_pool=False,
        caveats=[
            "本轮使用独立隔离重选的真人类洗发水原商品 B09B63YYCW（BELLISSO Biotin Shampoo）替代原误选的洗手液 B08XYBS678（Dove Hand Wash），目标商品保持 Aveda B00BDJODWI 不变。",
            "使用原始像素优化方法 Lucid（三个视觉代理 ViT-B-16 openai + ViT-B-32 openai + ViT-B-32 laion2b_s34b_b79k 联合优化，局部 Sinkhorn 最优传输权重 0.2，目标特点文字对齐 text_weight=1.0），在独立目录以 epsilon=8/255、1000 步完整从头生成。",
            "评测阶段在复制的 189 条共享长期记忆池隔离副本中，仅替换 07_shampoo 原条目为 B09B63YYCW 模拟参考条目（ObservedPurchase=false, RecordKind=simulated_gallery），其余 188 条及原有数据文件保持严格只读不变。",
            "同时评测 multimodal joint（图文各半，复用 20261110 独立模糊照片与 20 条特点句）与 text（纯文字，复用 20261109 同 20 条特点句）；仅统计检索排名，不执行购物。",
            "训练代理之一与指定评估编码器均为 ViT-B-32 (openai)，属于同编码器评估而非独立留出黑盒迁移；正式 20 条特点问句中有 1 条与唯一目标对齐文字精确重合。",
        ],
    )
    freeze_json(run / "EXPERIMENT_PLAN.json", plan)
    freeze_json(
        run / "PREPARED.json",
        dict(
            status="prepared",
            plan_sha256=sha256_file(run / "EXPERIMENT_PLAN.json"),
            formal_queries_read=False,
            shared_pool_read=False,
            proposed_pool_entry_read=False,
            shopping_enabled=False,
        ),
    )
    return plan


def validate_plan(run: Path) -> dict:
    plan = read_json(run / "EXPERIMENT_PLAN.json")
    opts = plan.get("generation_options", {})
    if (
        plan.get("group") != GROUP_KEY
        or plan.get("groups") != [GROUP_KEY]
        or plan.get("method") != METHOD
        or opts.get("steps") != REQUESTED_STEPS
        or opts.get("epsilon_levels") != EPSILON_LEVELS
        or float(opts.get("text_weight", -1.0)) != TEXT_WEIGHT
        or plan.get("shopping_enabled") is not False
    ):
        raise ValueError("实验计划范围、Lucid 超参数或停止购物标记发生篡改")
    for path, expected in plan["implementation"].items():
        _check(path, expected)
    for item in plan["generation_inputs"].values():
        for path, expected in item["files"].items():
            _check(path, expected)
    return plan


# ==============================================================================
# Stage 2: generate_lucid (epsilon=8/255, steps=1000) & freeze_images
# ==============================================================================


def method_dir(run: Path) -> Path:
    return run / "generation" / METHOD / GROUP_KEY


def validate_lucid_image(run: Path, plan: dict | None = None) -> dict:
    from src.visual.attack_completion import validate_attack_completion

    plan = plan or read_json(run / "EXPERIMENT_PLAN.json")
    item = plan["generation_inputs"][GROUP_KEY]
    opts = plan["generation_options"]
    m_root = method_dir(run)
    gen_dir = m_root / "generation"
    frozen_json_path = m_root / "FROZEN_IMAGE.json"
    frozen_png_path = m_root / "frozen.png"
    frozen_doc = read_json(frozen_json_path)
    meta = read_json(gen_dir / "generation_result.json")

    source_sha = sha256_file(item["source_image"])
    target_sha = sha256_file(item["target_image"])

    validate_attack_completion(
        meta,
        requested_steps=REQUESTED_STEPS,
        expected={
            "method": METHOD,
            "source_image_sha256": source_sha,
            "target_image_sha256": target_sha,
        },
    )

    attack = meta.get("attack", {})
    audit = meta.get("decoded_image_audit", {})
    txt_align = meta.get("text_alignment", {})
    if (
        abs(float(attack.get("epsilon", 0.0)) - EPSILON_LEVELS / 255.0) > 1e-9
        or float(attack.get("text_weight", -1.0)) != TEXT_WEIGHT
        or not audit.get("within_budget")
        or audit.get("budget_levels") != EPSILON_LEVELS
        or int(audit.get("max_absolute_pixel_difference_levels", 999)) > EPSILON_LEVELS
        or not txt_align.get("enabled")
        or txt_align.get("text_sha256") != hashlib.sha256(opts["target_text"].encode("utf-8")).hexdigest()
    ):
        raise ValueError("Lucid 完成成图预算或目标文字对齐配置与要求 (8/255, 1000步) 不符")

    # 独立校验实际落盘图片对原图的 L_inf 差异
    disk_audit = decoded_image_audit(item["source_image"], frozen_png_path, maximum_levels=EPSILON_LEVELS)
    if not disk_audit["within_budget"]:
        raise ValueError(f"冻结 Lucid 图片像素预算超限：{disk_audit}")

    if frozen_doc.get("status") != "frozen" or frozen_doc.get("image_sha256") != sha256_file(frozen_png_path):
        raise ValueError("Lucid 冻结图摘要不符")
    if sha256_file(frozen_png_path) != sha256_file(gen_dir / "adversarial.png"):
        raise ValueError("Lucid 冻结副本与原始生成图不一致")

    return dict(
        image=str(frozen_png_path.resolve()),
        raw_adversarial=str((gen_dir / "adversarial.png").resolve()),
        sha256=sha256_file(frozen_png_path),
        ready=str((gen_dir / "generation_result.json").resolve()),
        ready_sha256=sha256_file(gen_dir / "generation_result.json"),
        frozen_json=str(frozen_json_path.resolve()),
        frozen_json_sha256=sha256_file(frozen_json_path),
    )


def generate_lucid(run: Path) -> None:
    from src.visual.attack_completion import validate_attack_completion
    from src.visual.catalog_training_guard import catalog_training_guard
    from src.visual.optimization import generate_attack

    plan = validate_plan(run)
    if (run / "PAUSE_REQUESTED").exists():
        raise InterruptedError("本轮已请求暂停")

    item = plan["generation_inputs"][GROUP_KEY]
    opts = plan["generation_options"]
    m_root = method_dir(run)
    gen_dir = m_root / "generation"
    frozen_json_path = m_root / "FROZEN_IMAGE.json"
    frozen_png_path = m_root / "frozen.png"

    if frozen_json_path.is_file() and frozen_png_path.is_file() and (gen_dir / "generation_result.json").is_file():
        validate_lucid_image(run, plan)
        print(json.dumps(dict(stage="lucid_already_completed", group=GROUP_KEY), ensure_ascii=False), flush=True)
        return

    m_root.mkdir(parents=True, exist_ok=True)
    gen_dir.mkdir(parents=True, exist_ok=True)

    source_img = Path(item["source_image"]).resolve()
    target_img = Path(item["target_image"]).resolve()
    pub_def = item["public_definition"]

    source_record = copy.deepcopy(pub_def["source"])
    source_record["image"] = str(source_img)
    source_record["image_sha256"] = sha256_file(source_img)
    target_record = copy.deepcopy(pub_def["target"])
    target_record["image"] = str(target_img)
    target_record["image_sha256"] = sha256_file(target_img)

    public_binding = dict(
        method=METHOD,
        group=GROUP_KEY,
        selection_sha256=sha256_file(RESELECT_PKG / "definition.json"),
        source=source_record,
        target=target_record,
        options=opts,
        shared_pool_scope="isolated_lucid_generation_no_pool_read",
        threat_model="black_box_provisional",
    )
    freeze_json(m_root / "PUBLIC_RUN.json", public_binding)

    tick = time.perf_counter()
    with catalog_training_guard(
        allowed_output_roots=[run, RESELECT_PKG],
        private_paths=_private_paths_for_training(run),
    ) as audit:
        adv_path, metadata = generate_attack(
            source=source_img,
            target=target_img,
            output_dir=gen_dir,
            method=METHOD,
            steps=REQUESTED_STEPS,
            epsilon_levels=EPSILON_LEVELS,
            device="cuda",
            optimization_config=opts["optimization_config"],
            target_text=opts["target_text"],
            text_weight=TEXT_WEIGHT,
        )

    elapsed = time.perf_counter() - tick
    freeze_json(
        m_root / "TRAINING_ACCESS_AUDIT.json",
        {k: v for k, v in audit.items() if k != "active"},
    )
    if audit.get("blocked_accesses"):
        raise PermissionError(f"训练访问守护检测到越权访问：{audit['blocked_accesses']}")

    shutil.copyfile(adv_path, frozen_png_path)
    validate_attack_completion(
        metadata,
        requested_steps=REQUESTED_STEPS,
        expected={
            "method": METHOD,
            "source_image_sha256": source_record["image_sha256"],
            "target_image_sha256": target_record["image_sha256"],
        },
    )

    frozen_hash = sha256_file(frozen_png_path)
    frozen_data = dict(
        schema_version=1,
        method=METHOD,
        status="frozen",
        image=str(frozen_png_path.resolve()),
        image_sha256=frozen_hash,
        source=source_record,
        target=target_record,
        metadata=metadata,
        requested_steps=REQUESTED_STEPS,
        epsilon_levels=EPSILON_LEVELS,
        text_weight=TEXT_WEIGHT,
        observed_seconds=elapsed,
        public_binding=public_binding,
        public_binding_sha256=digest(public_binding),
    )
    freeze_json(frozen_json_path, frozen_data)
    validate_lucid_image(run, plan)
    print(
        json.dumps(
            dict(
                stage="lucid_generation_completed",
                group=GROUP_KEY,
                sha256=frozen_hash,
                max_diff_levels=metadata["decoded_image_audit"]["max_absolute_pixel_difference_levels"],
                seconds=round(elapsed, 2),
            ),
            ensure_ascii=False,
        ),
        flush=True,
    )


def freeze_images(run: Path) -> dict:
    plan = validate_plan(run)
    lucid_asset = validate_lucid_image(run, plan)
    payload = dict(
        status="frozen",
        images={
            METHOD: {GROUP_KEY: lucid_asset},
        },
        plan_sha256=sha256_file(run / "EXPERIMENT_PLAN.json"),
        formal_rankings_used=False,
    )
    freeze_json(run / "FROZEN_IMAGES.json", payload)
    return payload


# ==============================================================================
# Stage 3-5: Isolated Pool Replacement, Evaluation (joint + text), Verification, Report
# ==============================================================================


def _frozen_generation(run: Path) -> dict:
    run = Path(run).resolve()
    plan_file = run / "EXPERIMENT_PLAN.json"
    plan = read_json(plan_file)
    opts = plan.get("generation_options", {})
    if (
        plan.get("groups") != [GROUP_KEY]
        or plan.get("method") != METHOD
        or opts.get("steps") != REQUESTED_STEPS
        or opts.get("epsilon_levels") != EPSILON_LEVELS
        or plan.get("shopping_enabled") is not False
    ):
        raise ValueError("实验计划范围或 Lucid 参数无效")

    frozen_file = run / "FROZEN_IMAGES.json"
    frozen = read_json(frozen_file)
    if frozen.get("status") != "frozen" or set(frozen.get("images", {})) != {METHOD}:
        raise ValueError("必须先冻结完整 1000 步 Lucid 成图后方可评测")
    if frozen.get("plan_sha256") != sha256_file(plan_file):
        raise ValueError("冻结图片登记未绑定当前实验计划")

    validate_lucid_image(run, plan)
    _check(frozen["images"][METHOD][GROUP_KEY]["image"], frozen["images"][METHOD][GROUP_KEY]["sha256"])
    return frozen


def build_isolated_banks(
    entries: list[dict],
    pool_path: Path | str | None,
    replacement_entry: dict,
    image_assets: dict[str, dict],
    *,
    old_memory_id: str = OLD_SHAMPOO_MEMORY_ID,
    old_asin: str = OLD_SHAMPOO_ASIN,
    resolve_paths: bool = True,
) -> dict[str, list[dict]]:
    """在 189 条共享池副本中，仅替换 07_shampoo 源条目为重选洗发水 B09B63YYCW，保留其余 188 条严格不变。"""
    if len(entries) != 189 or len({e["Id"] for e in entries}) != 189:
        raise ValueError(f"必须使用完整且唯一的 189 条共享池，实际：{len(entries)}")

    old_indices = [idx for idx, entry in enumerate(entries) if entry["Id"] == old_memory_id]
    if len(old_indices) != 1:
        raise ValueError(f"共享池中未唯一找到待替换旧条目 {old_memory_id}：命中 {len(old_indices)} 条")
    old_idx = old_indices[0]
    old_entry = entries[old_idx]
    if old_entry.get("FinalASIN") != old_asin:
        raise ValueError(f"待替换条目 ASIN 不符：期望 {old_asin}，实际 {old_entry.get('FinalASIN')}")

    if set(replacement_entry.keys()) != set(old_entry.keys()):
        raise ValueError("替代条目与原 189 共享池条目顶级字段结构不一致")
    if set(replacement_entry["PrimaryImage"].keys()) != set(old_entry["PrimaryImage"].keys()):
        raise ValueError("替代条目 PrimaryImage 字段结构不一致")

    new_id = replacement_entry["Id"]
    new_asin = replacement_entry["FinalASIN"]
    if replacement_entry["PrimaryImage"].get("ASIN") != new_asin:
        raise ValueError("替代条目 PrimaryImage.ASIN 与 FinalASIN 不一致")
    if replacement_entry.get("RecordKind") != "simulated_gallery":
        raise ValueError("新洗发水参考条目必须如实标记 RecordKind=simulated_gallery")
    if (
        replacement_entry.get("ObservedPurchase") is not False
        or replacement_entry.get("Completed") is not False
        or replacement_entry.get("Actions") != []
    ):
        raise ValueError("未观察到购买的参考条目不得伪造 Completed 或 Actions 购买轨迹")

    remaining_ids = {e["Id"] for idx, e in enumerate(entries) if idx != old_idx}
    remaining_asins = {e.get("FinalASIN") for idx, e in enumerate(entries) if idx != old_idx}
    remaining_hashes = {(e.get("PrimaryImage") or {}).get("SHA256") for idx, e in enumerate(entries) if idx != old_idx}
    if new_id in remaining_ids or new_asin in remaining_asins or replacement_entry["PrimaryImage"]["SHA256"] in remaining_hashes:
        raise ValueError("替代条目与其余 188 条共享池条目发生 Id/ASIN/图片哈希冲突")

    clean = copy.deepcopy(entries)
    new_clean = copy.deepcopy(replacement_entry)

    for idx, entry in enumerate(clean):
        if entry.get("RetrievalImage") is not None or entry.get("ImageSubstitution") is not None:
            raise ValueError(f"基础共享池条目 {entry['Id']} 包含历史图片替换残留")
        if idx == old_idx:
            continue
        if resolve_paths and pool_path is not None:
            from src.visual.visual_retrieval import resolve_image
            img = resolve_image(entry, pool_path)
            _check(img, entry["PrimaryImage"]["SHA256"])
            entry["PrimaryImage"]["LocalPath"] = str(Path(img).resolve())
        if entry.get("RecordKind") == "simulated_gallery" and (entry.get("Completed") is True or entry.get("Actions")):
            raise ValueError(f"模拟参考条目 {entry['Id']} 不得含伪造购买动作")

    if resolve_paths:
        src_img = Path(new_clean["PrimaryImage"]["LocalPath"]).resolve()
        _check(src_img, new_clean["PrimaryImage"]["SHA256"])
        new_clean["PrimaryImage"]["LocalPath"] = str(src_img)
    clean[old_idx] = new_clean

    if len({e["Id"] for e in clean}) != 189:
        raise ValueError("替换后共享池条目 Id 数量异常")

    expected_modified = (METHOD,)
    if set(image_assets.keys()) != set(expected_modified):
        raise ValueError(f"修改图资产映射不符：期望 {expected_modified}，实际 {tuple(image_assets.keys())}")

    banks: dict[str, list[dict]] = {"clean": clean}
    for name in expected_modified:
        asset = image_assets[name]
        if resolve_paths:
            _check(asset["image"], asset["sha256"])
        bank = copy.deepcopy(clean)
        target_entry = bank[old_idx]
        if target_entry["Id"] != new_id:
            raise AssertionError("替换槽位 Id 不一致")
        target_entry["RetrievalImage"] = dict(
            ASIN=new_asin,
            LocalPath=str(Path(asset["image"]).resolve()),
            SHA256=asset["sha256"],
        )
        stripped = [{k: v for k, v in e.items() if k != "RetrievalImage"} for e in bank]
        if stripped != clean:
            raise ValueError(f"实验库 {name} 意外修改了检索图片以外的字段")
        replaced_ids = [e["Id"] for e in bank if "RetrievalImage" in e]
        if replaced_ids != [new_id]:
            raise ValueError(f"实验库 {name} 仅允许替换 {new_id} 一条图片，实际：{replaced_ids}")
        banks[name] = bank
    return banks


def _prepare_inputs(run: Path) -> dict:
    from src.agent.memory.shared_pool import load_shared_pool
    from src.experiments.query_contract import load_formal_queries
    from src.visual._perlage_ambient_coop import build_source_public_key
    from src.visual.visual_retrieval import detect_index_text_policy, memory_index_text

    frozen = _frozen_generation(run)
    plan = read_json(run / "EXPERIMENT_PLAN.json")
    gen_item = plan["generation_inputs"][GROUP_KEY]

    base_manifest = read_json(BASE / "manifest.json")
    pool_path = (BASE / base_manifest["shared_memory"]["pool_file"]).resolve()
    _check(pool_path, base_manifest["shared_memory"]["pool_sha256"])
    entries = load_shared_pool(pool_path)

    encoder_path = (BASE / base_manifest["evaluation_protocol"]["encoder_config"]).resolve()
    _check(encoder_path, base_manifest["evaluation_protocol"]["encoder_config_sha256"])

    proposed_path = (RESELECT_COMP / "proposed_pool_error_entry.json").resolve()
    replacement_entry = read_json(proposed_path)

    source_title = gen_item["public_definition"]["data"]["source_title"]
    expected_public_key = build_source_public_key(gen_item["source_public_record"], source_title)
    indexed_key = memory_index_text(replacement_entry, "product_feature")
    if indexed_key != expected_public_key:
        raise ValueError("新洗发水共享池特征检索键与公开源商品特点键不一致")

    bindings = {
        str((BASE / "manifest.json").resolve()): sha256_file(BASE / "manifest.json"),
        str(pool_path): sha256_file(pool_path),
        str(encoder_path): sha256_file(encoder_path),
        str(proposed_path): sha256_file(proposed_path),
        str((run / "EXPERIMENT_PLAN.json").resolve()): sha256_file(run / "EXPERIMENT_PLAN.json"),
        str((run / "FROZEN_IMAGES.json").resolve()): sha256_file(run / "FROZEN_IMAGES.json"),
        str((RESELECT_PKG / "source.jpg").resolve()): sha256_file(RESELECT_PKG / "source.jpg"),
        str((RESELECT_PKG / "target.jpg").resolve()): sha256_file(RESELECT_PKG / "target.jpg"),
    }

    lucid_asset = frozen["images"][METHOD][GROUP_KEY]
    bindings[lucid_asset["image"]] = lucid_asset["sha256"]
    bindings[lucid_asset["ready"]] = lucid_asset["ready_sha256"]

    banks = build_isolated_banks(
        entries,
        pool_path,
        replacement_entry,
        image_assets={
            METHOD: {"image": lucid_asset["image"], "sha256": lucid_asset["sha256"]},
        },
        resolve_paths=True,
    )

    bank_paths = {}
    for name in BANKS:
        path = run / "banks" / f"{name}.json"
        freeze_json(path, banks[name])
        bank_paths[name] = dict(path=str(path.resolve()), sha256=sha256_file(path))
        bindings[str(path.resolve())] = sha256_file(path)

    target_binding = {
        "id": TARGET_ASIN,
        "image": str((RESELECT_PKG / "target.jpg").resolve()),
        "image_sha256": sha256_file(RESELECT_PKG / "target.jpg"),
    }
    text_manifest = (BASE / base_manifest["groups"][GROUP_KEY]["query_manifest"]).resolve()
    _check(text_manifest, base_manifest["groups"][GROUP_KEY]["query_manifest_sha256"])
    joint_manifest = (QUERY_MULTIMODAL_RUN / "input_resources/formal_queries" / GROUP_KEY / "queries.json").resolve()
    if not joint_manifest.is_file():
        raise FileNotFoundError(f"缺少正式图文查询清单：{joint_manifest}")

    text_doc = read_json(text_manifest)
    joint_doc = read_json(joint_manifest)
    if joint_doc.get("instructions") != text_doc.get("instructions"):
        raise ValueError("joint 与 text 查询集的二十条特点问句正文不一致")

    queries_by_mode = {
        "joint": load_formal_queries(joint_manifest, target_binding, query_mode="joint"),
        "text": load_formal_queries(text_manifest, target_binding, query_mode="text"),
    }
    for mode, qlist in queries_by_mode.items():
        if len(qlist) != 20:
            raise ValueError(f"{mode} 模式正式请求数须为 20，实际 {len(qlist)}")
        for q in qlist:
            if q.get("query_image"):
                bindings[q["query_image"]] = q["query_image_sha256"]
    bindings[str(text_manifest)] = sha256_file(text_manifest)
    bindings[str(joint_manifest)] = sha256_file(joint_manifest)

    expected_rankings = len(QUERY_MODES) * len(BANKS) * 20
    expected_score_comparisons = expected_rankings * 189

    inputs = dict(
        schema_version=1,
        group=GROUP_KEY,
        method=METHOD,
        source_id=NEW_SHAMPOO_ASIN,
        target_id=TARGET_ASIN,
        replaced_memory_id=OLD_SHAMPOO_MEMORY_ID,
        source_memory_id=NEW_SHAMPOO_MEMORY_ID,
        source_public_key_text=indexed_key,
        source_public_key_sha256=hashlib.sha256(indexed_key.encode("utf-8")).hexdigest(),
        bank_order=list(BANKS),
        query_modes=list(QUERY_MODES),
        pool_count=189,
        record_kinds={
            kind: sum(e["RecordKind"] == kind for e in banks["clean"])
            for kind in ("purchase_memory", "simulated_gallery")
        },
        banks=bank_paths,
        queries_by_mode=queries_by_mode,
        training_overlap=joint_doc.get("training_overlap"),
        encoder_config=str(encoder_path),
        encoder_config_sha256=sha256_file(encoder_path),
        bindings=bindings,
        query_image_weight={"joint": 0.5, "text": 0.0},
        key_image_weight=0.5,
        index_text=detect_index_text_policy(banks["clean"]),
        condition="none",
        shopping_allowed=False,
        expected_rankings=expected_rankings,
        expected_score_comparisons=expected_score_comparisons,
        persistent_vector_cache_used=False,
    )
    freeze_json(run / "EVALUATION_INPUTS.json", inputs)
    return inputs


def _inputs(run: Path) -> dict:
    _frozen_generation(run)
    inputs = read_json(run / "EVALUATION_INPUTS.json")
    if (
        inputs.get("bank_order") != list(BANKS)
        or inputs.get("query_modes") != list(QUERY_MODES)
        or inputs.get("pool_count") != 189
        or inputs.get("shopping_allowed") is not False
    ):
        raise ValueError("冻结评测范围或停止购物约束发生变更")
    for path, expected in inputs["bindings"].items():
        _check(path, expected)
    return inputs


class _MemoizedEncoder:
    """仅在当前独立进程内存中复用编解码结果，绝不读/写磁盘向量缓存。"""

    supports_image_weight = True

    def __init__(self, encoder):
        self.encoder = encoder
        self.snapshot = encoder.snapshot
        self.cache: dict = {}
        self.last_encoding_audit = None

    def encode_text(self, text: str):
        key = ("text", hashlib.sha256(text.encode("utf-8")).hexdigest())
        if key not in self.cache:
            vector = self.encoder.encode_text(text)
            self.cache[key] = (vector, copy.deepcopy(getattr(self.encoder, "last_encoding_audit", None)))
        vector, audit = self.cache[key]
        self.last_encoding_audit = copy.deepcopy(audit)
        return vector

    def encode_joint(self, text: str, image: str | Path, image_weight: float = 0.5):
        key = (
            "joint",
            hashlib.sha256(text.encode("utf-8")).hexdigest(),
            sha256_file(image),
            float(image_weight),
        )
        if key not in self.cache:
            vector = (
                self.encoder.encode_joint(text, image)
                if image_weight == 0.5
                else self.encoder.encode_joint(text, image, image_weight=image_weight)
            )
            self.cache[key] = (vector, copy.deepcopy(getattr(self.encoder, "last_encoding_audit", None)))
        vector, audit = self.cache[key]
        self.last_encoding_audit = copy.deepcopy(audit)
        return vector


def _result_path(run: Path, mode: str, bank: str) -> Path:
    return run / "retrieval" / mode / f"{bank}.json"


def _metrics(ranks: list[int]) -> dict:
    count = len(ranks)
    if not count or any(type(r) is not int or not 1 <= r <= 189 for r in ranks):
        raise ValueError("排名范围无效")
    return dict(
        query_count=count,
        min_rank=min(ranks),
        max_rank=max(ranks),
        mean_rank=sum(ranks) / count,
        mrr=sum(1.0 / r for r in ranks) / count,
        **{f"top{k}": sum(r <= k for r in ranks) for k in (1, 3, 5)},
        **{f"top{k}_rate": sum(r <= k for r in ranks) / count for k in (1, 3, 5)},
    )


def _ranker(encoder, inputs: dict, mode: str, bank_name: str):
    from src.visual.visual_retrieval import VisualMemoryRetriever

    bank_path = inputs["banks"][bank_name]["path"]
    bank = read_json(bank_path)
    if len(bank) != 189:
        raise ValueError(f"检索库 {bank_name} 条目数不为 189")
    q_weight = inputs["query_image_weight"][mode]
    ranker = VisualMemoryRetriever(
        encoder,
        bank_path,
        cache_dir=None,
        key_mode="joint",
        query_mode=mode,
        index_text=inputs["index_text"],
        key_image_weight=inputs["key_image_weight"],
        query_image_weight=q_weight,
    )
    return bank, ranker


def _validate_result(run: Path, inputs: dict, mode: str, bank_name: str) -> dict:
    result = read_json(_result_path(run, mode, bank_name))
    protocol = result.get("protocol", {})
    exp_q_weight = inputs["query_image_weight"][mode]
    if (
        result.get("inputs_sha256") != sha256_file(run / "EVALUATION_INPUTS.json")
        or result.get("mode") != mode
        or result.get("bank") != bank_name
        or result.get("group") != GROUP_KEY
        or protocol.get("query_mode") != mode
        or protocol.get("key_mode") != "joint"
        or protocol.get("key_image_weight") != inputs["key_image_weight"]
        or protocol.get("query_image_weight") != exp_q_weight
        or protocol.get("bank_sha256") != inputs["banks"][bank_name]["sha256"]
    ):
        raise ValueError(f"检索结果协议与输入绑定不匹配：{mode}/{bank_name}")

    queries = inputs["queries_by_mode"][mode]
    rows = result.get("rankings", [])
    bank_ids = {e["Id"] for e in read_json(inputs["banks"][bank_name]["path"])}
    if len(rows) != 20 or [r.get("task_id") for r in rows] != [q["task_id"] for q in queries]:
        raise ValueError(f"检索结果请求数量或任务序列不匹配：{mode}/{bank_name}")

    for row in rows:
        full = row.get("ranking", [])
        if (
            len(full) != 189
            or {r["Id"] for r in full} != bank_ids
            or [r["rank"] for r in full] != list(range(1, 190))
            or any(not math.isfinite(r["score"]) for r in full)
            or any(a["score"] < b["score"] for a, b in zip(full, full[1:]))
        ):
            raise ValueError(f"全库 189 条候选有序排名检验失败：{mode}/{bank_name}")
        source = next(r for r in full if r["Id"] == inputs["source_memory_id"])
        if row.get("source_rank") != source["rank"] or row.get("source_score") != source["score"]:
            raise ValueError(f"源条目摘要名次/分数与完整排名不一致：{mode}/{bank_name}")

    if result.get("metrics") != _metrics([row["source_rank"] for row in rows]):
        raise ValueError(f"汇总指标与逐请求名次不符：{mode}/{bank_name}")
    return result


def _validate_retrieval(run: Path, inputs: dict) -> dict:
    cert = read_json(run / "RETRIEVAL_COMPLETED.json")
    expected_files = {str(_result_path(run, mode, bank).resolve()) for mode in QUERY_MODES for bank in BANKS}
    if (
        cert.get("status") != "completed"
        or cert.get("rankings") != inputs["expected_rankings"]
        or cert.get("score_count") != inputs["expected_score_comparisons"]
        or cert.get("execution_device") != "cuda"
        or cert.get("inputs_sha256") != sha256_file(run / "EVALUATION_INPUTS.json")
        or set(cert.get("files", {})) != expected_files
    ):
        raise ValueError("检索完成证书范围或绑定不完整")
    for path, expected in cert["files"].items():
        _check(path, expected)
    for mode in QUERY_MODES:
        for bank in BANKS:
            _validate_result(run, inputs, mode, bank)
    return cert


def _validate_verification(run: Path, inputs: dict) -> dict:
    retrieval = _validate_retrieval(run, inputs)
    ver = read_json(run / "INDEPENDENT_VERIFICATION.json")
    if (
        ver.get("status") != "passed"
        or ver.get("score_comparisons") != inputs["expected_score_comparisons"]
        or ver.get("rankings_recomputed") != inputs["expected_rankings"]
        or ver.get("fresh_process") is not True
        or ver.get("fresh_encoder") is not True
        or ver.get("execution_device") != "cuda"
        or ver.get("pid") == retrieval.get("pid")
        or ver.get("retrieval_pid") != retrieval.get("pid")
        or ver.get("files") != retrieval.get("files")
        or ver.get("inputs_sha256") != sha256_file(run / "EVALUATION_INPUTS.json")
        or not isinstance(ver.get("max_score_error"), (int, float))
        or not math.isfinite(ver["max_score_error"])
        or ver["max_score_error"] > 1e-6
    ):
        raise ValueError("独立复算证书未通过校验")
    _check(run / "RETRIEVAL_COMPLETED.json", ver["retrieval_certificate_sha256"])
    return ver


def evaluate(run: Path | str) -> None:
    """Lucid 完整成图冻结后，新建编码器执行 joint 与 text 双模式全库 189 条零磁盘缓存检索评测。"""
    run = Path(run).resolve()
    inputs = _inputs(run) if (run / "EVALUATION_INPUTS.json").is_file() else _prepare_inputs(run)
    if (run / "RETRIEVAL_COMPLETED.json").is_file():
        _validate_retrieval(run, inputs)
        return

    from src.experiments.query_contract import verify_query_input
    from src.experiments.shared_evaluation import make_eval_encoder

    raw = make_eval_encoder("cuda", inputs["encoder_config"])
    encoder = _MemoizedEncoder(raw)
    try:
        for mode in QUERY_MODES:
            for bank_name in BANKS:
                out_path = _result_path(run, mode, bank_name)
                if out_path.exists():
                    _validate_result(run, inputs, mode, bank_name)
                    continue
                bank, ranker = _ranker(encoder, inputs, mode, bank_name)
                rows = []
                for query in inputs["queries_by_mode"][mode]:
                    verify_query_input(query)
                    ranking = ranker.rank(bank, query["instruction"], query.get("query_image"))
                    if len(ranking) != 189 or any(not math.isfinite(r["score"]) for r in ranking):
                        raise ValueError("完整检索分数无效")
                    source = next(r for r in ranking if r["Id"] == inputs["source_memory_id"])
                    rows.append(
                        dict(
                            task_id=query["task_id"],
                            source_rank=source["rank"],
                            source_score=source["score"],
                            ranking=ranking,
                        )
                    )
                freeze_json(
                    out_path,
                    dict(
                        mode=mode,
                        bank=bank_name,
                        group=GROUP_KEY,
                        source_asin=inputs["source_id"],
                        source_memory_id=inputs["source_memory_id"],
                        protocol=copy.deepcopy(ranker.snapshot),
                        rankings=rows,
                        metrics=_metrics([r["source_rank"] for r in rows]),
                        inputs_sha256=sha256_file(run / "EVALUATION_INPUTS.json"),
                    ),
                )
                m = _metrics([r["source_rank"] for r in rows])
                print(
                    f"检索完成 [{mode}/{bank_name}]：Top-1 {m['top1']}/20, Top-3 {m['top3']}/20, Top-5 {m['top5']}/20, MeanRank {m['mean_rank']:.3f}",
                    flush=True,
                )
    finally:
        raw.close()

    files = {str(_result_path(run, mode, bank).resolve()): sha256_file(_result_path(run, mode, bank)) for mode in QUERY_MODES for bank in BANKS}
    freeze_json(
        run / "RETRIEVAL_COMPLETED.json",
        dict(
            status="completed",
            pid=os.getpid(),
            execution_device="cuda",
            inputs_sha256=sha256_file(run / "EVALUATION_INPUTS.json"),
            rankings=inputs["expected_rankings"],
            score_count=inputs["expected_score_comparisons"],
            persistent_vector_cache_read=False,
            files=files,
        ),
    )


def verify(run: Path | str) -> None:
    """严格在独立子进程从零重编码全部 189 条候选与查询，逐项比对全部 15,120 个分数。"""
    run = Path(run).resolve()
    inputs = _inputs(run)
    retrieval = _validate_retrieval(run, inputs)
    if (run / "INDEPENDENT_VERIFICATION.json").is_file():
        _validate_verification(run, inputs)
        return

    if retrieval.get("pid") == os.getpid():
        raise ValueError("独立复算必须使用与主评测不同的独立子进程")
    if retrieval.get("inputs_sha256") != sha256_file(run / "EVALUATION_INPUTS.json"):
        raise ValueError("检索证书未绑定本轮评估输入")
    for path, expected in retrieval["files"].items():
        _check(path, expected)

    from src.experiments.query_contract import verify_query_input
    from src.experiments.shared_evaluation import make_eval_encoder

    raw = make_eval_encoder("cuda", inputs["encoder_config"])
    encoder = _MemoizedEncoder(raw)
    compared, rankings, max_err = 0, 0, 0.0
    try:
        for mode in QUERY_MODES:
            for bank_name in BANKS:
                bank, ranker = _ranker(encoder, inputs, mode, bank_name)
                old = read_json(_result_path(run, mode, bank_name))
                if digest(encoder.snapshot) != old["protocol"]["model_sha256"] or len(old["rankings"]) != 20:
                    raise ValueError("独立复算编码器快照或请求数量发生变化")
                for query, prev in zip(inputs["queries_by_mode"][mode], old["rankings"]):
                    verify_query_input(query)
                    actual = ranker.rank(bank, query["instruction"], query.get("query_image"))
                    if query["task_id"] != prev["task_id"] or len(actual) != len(prev["ranking"]) or len(actual) != 189:
                        raise ValueError("独立复算请求序号或候选数不一致")
                    for a, b in zip(actual, prev["ranking"]):
                        if any(
                            a[f] != b[f]
                            for f in ("Id", "rank", "image_sha256", "text_sha256", "query_text_sha256", "query_image_sha256")
                        ):
                            raise ValueError(f"独立复算候选字段或排序不一致：{mode}/{bank_name}/{query['task_id']}")
                        err = abs(a["score"] - b["score"])
                        if not math.isfinite(err) or err > 1e-6:
                            raise ValueError(f"独立复算分数差异超限：{err:.9g} > 1e-6")
                        compared += 1
                        max_err = max(max_err, err)
                    src = next(r for r in actual if r["Id"] == inputs["source_memory_id"])
                    if src["rank"] != prev["source_rank"] or abs(src["score"] - prev["source_score"]) > 1e-6:
                        raise ValueError("独立复算源条目名次或分数与记录不一致")
                    rankings += 1
                if old["metrics"] != _metrics([r["source_rank"] for r in old["rankings"]]):
                    raise ValueError("既有指标与排名不一致")
    finally:
        raw.close()

    if compared != inputs["expected_score_comparisons"] or rankings != inputs["expected_rankings"]:
        raise ValueError(f"独立复算未覆盖预期全量：{rankings}/{inputs['expected_rankings']}, {compared}/{inputs['expected_score_comparisons']}")

    freeze_json(
        run / "INDEPENDENT_VERIFICATION.json",
        dict(
            status="passed",
            fresh_process=True,
            pid=os.getpid(),
            retrieval_pid=retrieval["pid"],
            fresh_encoder=True,
            execution_device="cuda",
            rankings_recomputed=rankings,
            score_comparisons=compared,
            max_score_error=max_err,
            persistent_vector_cache_read=False,
            inputs_sha256=sha256_file(run / "EVALUATION_INPUTS.json"),
            retrieval_certificate_sha256=sha256_file(run / "RETRIEVAL_COMPLETED.json"),
            files=retrieval["files"],
        ),
    )


def _paired(current: list[dict], previous: list[dict]) -> dict:
    if [r["task_id"] for r in current] != [r["task_id"] for r in previous]:
        raise ValueError("配对名次正式请求标识不一致")
    pairs = [(a["source_rank"], b["source_rank"]) for a, b in zip(current, previous)]
    return dict(
        top1_gained=sum(a <= 1 < b for a, b in pairs),
        top1_lost=sum(b <= 1 < a for a, b in pairs),
        top3_gained=sum(a <= 3 < b for a, b in pairs),
        top3_lost=sum(b <= 3 < a for a, b in pairs),
        top5_gained=sum(a <= 5 < b for a, b in pairs),
        top5_lost=sum(b <= 5 < a for a, b in pairs),
        improved_ranks=sum(a < b for a, b in pairs),
        worsened_ranks=sum(a > b for a, b in pairs),
        tied_ranks=sum(a == b for a, b in pairs),
    )


def _compute_lucid_visual(run: Path, plan: dict) -> dict:
    item = plan["generation_inputs"][GROUP_KEY]
    source_path = Path(item["source_image"])
    m_root = method_dir(run)
    gen_dir = m_root / "generation"
    meta = read_json(gen_dir / "generation_result.json")
    final_path = m_root / "frozen.png"

    source_rgb = read_rgb(source_path)
    final_rgb = read_rgb(final_path)
    diff = np.abs(final_rgb.astype(np.float32) - source_rgb.astype(np.float32))
    disk_audit = decoded_image_audit(source_path, final_path, maximum_levels=EPSILON_LEVELS)

    t_final = torch.as_tensor(final_rgb.transpose(2, 0, 1), dtype=torch.float32)[None] / 255.0
    t_source = torch.as_tensor(source_rgb.transpose(2, 0, 1), dtype=torch.float32)[None] / 255.0

    # 分区视觉分析（使用公开前景掩码和印刷区域框，仅作报告统计）
    fg_mask_path = Path(source_path).parent / item["public_definition"]["source"]["foreground_mask"]
    if fg_mask_path.is_file():
        fg_mask = read_rgb(fg_mask_path)[..., 0] > 0
        fg_mae = float(diff[fg_mask].mean()) if fg_mask.any() else 0.0
        bg_mae = float(diff[~fg_mask].mean()) if (~fg_mask).any() else 0.0
    else:
        fg_mae = None
        bg_mae = None

    roi_mask = print_region_mask(source_rgb.shape, item["public_definition"]["print_regions"])
    roi_diff = diff[roi_mask] if roi_mask.any() else np.array([0.0], dtype=np.float32)
    mse = float(disk_audit["mean_squared_pixel_difference_levels"])
    psnr = float(20.0 * math.log10(255.0 / math.sqrt(mse))) if mse > 0.0 else float("inf")

    attack = meta.get("attack", {})
    saved_obj = meta.get("after_attack_saved") or meta.get("final_objective") or {}
    return dict(
        method=METHOD,
        steps_requested=int(attack.get("steps_requested", REQUESTED_STEPS)),
        steps_completed=int(attack.get("steps_completed", REQUESTED_STEPS)),
        selected_step=int(attack.get("selected_step", REQUESTED_STEPS)),
        epsilon_levels=EPSILON_LEVELS,
        epsilon=float(attack.get("epsilon", EPSILON_LEVELS / 255.0)),
        text_weight=float(attack.get("text_weight", TEXT_WEIGHT)),
        source_ssim=float(source_ssim(t_final, t_source)),
        max_abs_diff_levels=int(disk_audit["max_absolute_pixel_difference_levels"]),
        source_mae_levels=float(disk_audit["mean_absolute_pixel_difference_levels"]),
        source_mse_levels=mse,
        source_psnr_db=psnr if math.isfinite(psnr) else None,
        fg_source_mae_levels=fg_mae,
        bg_source_mae_levels=bg_mae,
        roi_source_max_levels=int(round(float(roi_diff.max()))),
        roi_source_mae_levels=float(roi_diff.mean()),
        after_attack_saved_loss=saved_obj.get("loss"),
        after_attack_text_similarity=saved_obj.get("text_similarity"),
        after_attack_global_similarities=saved_obj.get("global_similarity"),
    )


def _save_visual_contact_sheet(run: Path, plan: dict) -> Path:
    item = plan["generation_inputs"][GROUP_KEY]
    src = Image.open(item["source_image"]).convert("RGB")
    lucid_img = Image.open(method_dir(run) / "frozen.png").convert("RGB")

    arr_src = np.asarray(src, dtype=np.int16)
    arr_lucid = np.asarray(lucid_img, dtype=np.int16)
    diff_x10 = Image.fromarray(np.clip(np.abs(arr_lucid - arr_src) * 10, 0, 255).astype(np.uint8))

    w, h = src.size
    pad = 16
    banner = 36
    panels = [
        ("1. Reselected Clean (B09B63YYCW)", src),
        ("2. Lucid Final (8/255, 1000s)", lucid_img),
        ("3. |Lucid - Clean| x10", diff_x10),
    ]
    canvas = Image.new("RGB", (len(panels) * w + (len(panels) + 1) * pad, h + banner + pad), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    for idx, (label, img) in enumerate(panels):
        x = pad + idx * (w + pad)
        draw.text((x, 10), label, fill=(20, 20, 20))
        canvas.paste(img, (x, banner))
    out = run / "comparison_visual_07_shampoo_lucid.png"
    canvas.save(out)
    return out


def _top_competitors(result_doc: dict, bank_lookup: dict[str, dict], query_idx: int = 0, limit: int = 5) -> list[dict]:
    row = result_doc["rankings"][query_idx]
    out = []
    for item in row["ranking"][:limit]:
        meta = bank_lookup[item["Id"]]
        out.append(
            dict(
                rank=item["rank"],
                id=item["Id"],
                asin=meta.get("FinalASIN", ""),
                title=meta["PrimaryImage"]["Title"],
                brand=meta.get("Brand", meta["PrimaryImage"].get("Brand", "")),
                score=item["score"],
                is_reselected_source=(item["Id"] == result_doc["source_memory_id"]),
            )
        )
    return out


def _load_readonly_comparisons() -> dict:
    """只读读取同一重选组 ambient_coop 及历史旧洗发水组 B08XYBS678 指标供完整对比展示。"""
    ref: dict = {}
    if (HIST_AMBIENT_RESELECT_RUN / "SUMMARY.json").is_file():
        rs = read_json(HIST_AMBIENT_RESELECT_RUN / "SUMMARY.json")
        ref["reselect_b09_other_methods"] = {
            "source_summary": str((HIST_AMBIENT_RESELECT_RUN / "SUMMARY.json").resolve()),
            "modes": {
                mode: {
                    bank: rs["modes"][mode][bank]["metrics"]
                    for bank in ("clean", "baseline", "ambient_coop")
                    if bank in rs.get("modes", {}).get(mode, {})
                }
                for mode in ("text", "joint")
            },
            "visual": rs.get("visual_metrics", {}),
        }

    hist_b08: dict = {}
    if (HIST_AMBIENT_RUN / "SUMMARY.json").is_file():
        s = read_json(HIST_AMBIENT_RUN / "SUMMARY.json")
        hist_b08["joint_b08"] = {
            bank: s["banks"][bank]["metrics"]["07_shampoo"]
            for bank in ("clean", "baseline", "lucid", "ambient_coop")
            if bank in s.get("banks", {})
        }
    if (HIST_SHAMPOO_MODES / "AUDIT.json").is_file():
        a = read_json(HIST_SHAMPOO_MODES / "AUDIT.json")
        lucid_audit = a.get("results", {}).get("lucid", {}).get("A_feature_text", {})
        no_pixel_audit = a.get("results", {}).get("perlage_no_pixel", {}).get("A_feature_text", {})
        text_b08: dict = {}
        if no_pixel_audit.get("clean"):
            text_b08["clean"] = no_pixel_audit["clean"]["stats"]
        if no_pixel_audit.get("modified"):
            text_b08["baseline"] = no_pixel_audit["modified"]["stats"]
        if lucid_audit.get("modified"):
            text_b08["lucid"] = lucid_audit["modified"]["stats"]
        if text_b08:
            hist_b08["text_b08"] = text_b08
    if hist_b08:
        ref["historical_b08"] = hist_b08
    return ref


def report(run: Path | str) -> dict:
    run = Path(run).resolve()
    plan = validate_plan(run)
    inputs = _inputs(run)
    verification = _validate_verification(run, inputs)
    clean_bank = {e["Id"]: e for e in read_json(inputs["banks"]["clean"]["path"])}

    results = {
        mode: {bank: read_json(_result_path(run, mode, bank)) for bank in BANKS}
        for mode in QUERY_MODES
    }
    visual = _compute_lucid_visual(run, plan)
    contact_sheet = _save_visual_contact_sheet(run, plan)

    modes_summary = {}
    for mode in QUERY_MODES:
        bank_entries = {}
        for bank in BANKS:
            rows = results[mode][bank]["rankings"]
            comparisons = {}
            for ref_bank in [b for b in BANKS if b != bank]:
                prev_rows = results[mode][ref_bank]["rankings"]
                before = results[mode][ref_bank]["metrics"]
                after = results[mode][bank]["metrics"]
                comparisons[ref_bank] = dict(
                    **_paired(rows, prev_rows),
                    mean_rank_reduction=before["mean_rank"] - after["mean_rank"],
                    mrr_gain=after["mrr"] - before["mrr"],
                    **{f"top{k}_gain": after[f"top{k}"] - before[f"top{k}"] for k in (1, 3, 5)},
                )
            bank_entries[bank] = dict(
                metrics=results[mode][bank]["metrics"],
                comparisons=comparisons,
                query01_top5=_top_competitors(results[mode][bank], clean_bank, query_idx=0, limit=5),
                per_query_source=[
                    dict(task_id=r["task_id"], rank=r["source_rank"], score=r["source_score"])
                    for r in rows
                ],
            )
        modes_summary[mode] = bank_entries

    readonly_comp = _load_readonly_comparisons()
    summary = dict(
        schema_version=1,
        status="completed",
        scope="retrieval_only",
        group=GROUP_KEY,
        method=METHOD,
        epsilon_levels=EPSILON_LEVELS,
        steps=REQUESTED_STEPS,
        source_id=inputs["source_id"],
        target_id=inputs["target_id"],
        replaced_memory_id=inputs["replaced_memory_id"],
        source_memory_id=inputs["source_memory_id"],
        pool_count=189,
        record_kinds=inputs["record_kinds"],
        query_modes=list(QUERY_MODES),
        query_count_per_mode=20,
        shopping_executed=False,
        inputs_sha256=sha256_file(run / "EVALUATION_INPUTS.json"),
        verification_sha256=sha256_file(run / "INDEPENDENT_VERIFICATION.json"),
        contact_sheet=str(contact_sheet.resolve()),
        contact_sheet_sha256=sha256_file(contact_sheet),
        visual_metrics=visual,
        modes=modes_summary,
        readonly_comparisons=readonly_comp,
    )
    freeze_json(run / "SUMMARY.json", summary)

    bank_labels = {
        "clean": "原图库 (clean, B09B63YYCW 原图)",
        "lucid": "Lucid (8/255, 1000步, text_weight=1.0)",
    }
    mode_labels = {
        "joint": "环境二：特点文字＋独立模糊照片 (joint, 图文各半)",
        "text": "环境一：特点纯文字 (text, 无查询图片)",
    }

    lines = [
        "# 重选洗发水组（B09B63YYCW -> B00BDJODWI）Lucid (8/255, 1000步) 生成与共享记忆检索报告",
        "",
        "## 一、实验范围与隔离约束核验",
        "",
        f"- **重选原商品**：`{inputs['source_id']}`（BELLISSO Biotin Shampoo，真人类洗发水），在 189 共享池隔离副本中替换 `{inputs['replaced_memory_id']}`（其余 188 条及只读基础库完全不变）。",
        f"- **固定目标商品**：`{inputs['target_id']}`（Aveda Scalp Benefits Balancing Shampoo 33.8 OZ），唯一文本对齐句：`{plan['generation_options']['target_text']}`。",
        f"- **方法超参数**：`method=lucid`, `epsilon=8/255 (8 灰阶)`, `steps=1000`, `text_weight=1.0`, 三视觉代理联合优化（`ViT-B-16 openai`, `ViT-B-32 openai`, `ViT-B-32 laion2b_s34b_b79k`）。",
        f"- **参考标记与购买约束**：新条目保留 `ObservedPurchase=false` 与 `RecordKind=simulated_gallery`（{summary['record_kinds']['purchase_memory']} 条真实购买记忆 + {summary['record_kinds']['simulated_gallery']} 条模拟参考）；`shopping_executed={summary['shopping_executed']}`。",
        "- **编码器口径**：训练代理之一与指定检索编码器均为 `ViT-B-32 (openai)`，属同编码器公开权重评测。",
        "",
        "## 二、共享长期记忆库检索排名结果（189 条完整库，20 条正式请求）",
        "",
    ]

    for mode in ("text", "joint"):
        lines += [
            f"### {mode_labels[mode]}",
            "",
            "| 图库 / 方案 | 请求数 | Top-1 | Top-3 | Top-5 | 平均名次↓ | 名次范围 | MRR↑ |",
            "|---|---:|---:|---:|---:|---:|:---:|---:|",
        ]
        for bank in BANKS:
            m = modes_summary[mode][bank]["metrics"]
            lines.append(
                f"| {bank_labels[bank]} | {m['query_count']} | "
                f"{m['top1']}/20（{m['top1_rate']:.2%}） | {m['top3']}/20（{m['top3_rate']:.2%}） | "
                f"{m['top5']}/20（{m['top5_rate']:.2%}） | {m['mean_rank']:.3f} | "
                f"{m['min_rank']}–{m['max_rank']} | {m['mrr']:.5f} |"
            )
        d = modes_summary[mode]["lucid"]["comparisons"]["clean"]
        lines += [
            "",
            f"- **配对增量（Lucid vs Clean）**：Top-1 `{d['top1_gain']:+d}`，Top-3 `{d['top3_gain']:+d}`，Top-5 `{d['top5_gain']:+d}`，改善/退化/持平 `{d['improved_ranks']} / {d['worsened_ranks']} / {d['tied_ranks']}`，平均名次改善 `{d['mean_rank_reduction']:+.3f}`，MRR 增量 `{d['mrr_gain']:+.5f}`。",
            "",
        ]

    lines += [
        "## 三、Lucid (8/255, 1000步) 外观与扰动预算审计",
        "",
        "| 指标 | 实测值 |",
        "|---|---:|",
        f"| 请求/完成步数（选择步） | {visual['steps_requested']} / {visual['steps_completed']}（第 {visual['selected_step']} 步） |",
        f"| L_inf 最大像素级差异（预算 ≤ 8 级） | {visual['max_abs_diff_levels']} 级 |",
        f"| 全图 SSIM↑ | {visual['source_ssim']:.4f} |",
        f"| 全图 MAE (级)↓ | {visual['source_mae_levels']:.2f} |",
        f"| 全图 MSE / PSNR (dB) | {visual['source_mse_levels']:.2f} / {visual['source_psnr_db']:.2f} dB |" if visual["source_psnr_db"] is not None else f"| 全图 MSE | {visual['source_mse_levels']:.2f} |",
        f"| 前景/背景/印刷区 MAE (级) | {visual['fg_source_mae_levels']:.2f} / {visual['bg_source_mae_levels']:.2f} / {visual['roi_source_mae_levels']:.2f}（印刷区 Max {visual['roi_source_max_levels']} 级） |" if visual["fg_source_mae_levels"] is not None else f"| 印刷区 MAE/Max (级) | {visual['roi_source_mae_levels']:.2f} / {visual['roi_source_max_levels']} 级 |",
        "",
        f"视觉拼接对比图已保存至 `{contact_sheet.name}`。",
        "",
    ]

    if readonly_comp:
        lines += [
            "## 四、Top-3 跨组跨方法完整对照表（严格只读汇总）",
            "",
            "| 实验环境 | 方法 | 旧组 `B08XYBS678` Top-3 | 本轮新组 `B09B63YYCW` Top-3 |",
            "|---|---|---:|---:|",
        ]
        hist_text = readonly_comp.get("historical_b08", {}).get("text_b08", {})
        hist_joint = readonly_comp.get("historical_b08", {}).get("joint_b08", {})
        other_modes = readonly_comp.get("reselect_b09_other_methods", {}).get("modes", {})

        def _fmt_old_text(k: str) -> str:
            st = hist_text.get(k)
            if not st:
                return "—"
            return f"{st['top3']}/20 ({st['top3'] / 20:.2%})"

        def _fmt_old_joint(k: str) -> str:
            st = hist_joint.get(k)
            if not st:
                return "—"
            return f"{st['top3']}/20 ({st['top3'] / 20:.2%})"

        def _fmt_new(mode: str, bank: str) -> str:
            if bank == "lucid":
                m = modes_summary[mode]["lucid"]["metrics"]
                return f"{m['top3']}/20 ({m['top3_rate']:.2%})"
            if bank == "clean":
                m = modes_summary[mode]["clean"]["metrics"]
                return f"{m['top3']}/20 ({m['top3_rate']:.2%})"
            m = other_modes.get(mode, {}).get(bank)
            if not m:
                return "—"
            return f"{m['top3']}/20 ({m['top3_rate']:.2%})"

        rows_spec = [
            ("环境一：纯特征文字", "Clean（未修改原图）", _fmt_old_text("clean"), _fmt_new("text", "clean")),
            ("环境一：纯特征文字", "PerLAGE_no_pixel", _fmt_old_text("baseline"), _fmt_new("text", "baseline")),
            ("环境一：纯特征文字", "ambient_coop（本轮新方法）", "—", _fmt_new("text", "ambient_coop")),
            ("环境一：纯特征文字", "Lucid（8/255, 1000步）", _fmt_old_text("lucid"), _fmt_new("text", "lucid")),
            ("环境二：特征文字 + 模糊图片", "Clean（未修改原图）", _fmt_old_joint("clean"), _fmt_new("joint", "clean")),
            ("环境二：特征文字 + 模糊图片", "PerLAGE_no_pixel", _fmt_old_joint("baseline"), _fmt_new("joint", "baseline")),
            ("环境二：特征文字 + 模糊图片", "ambient_coop（本轮新方法）", _fmt_old_joint("ambient_coop"), _fmt_new("joint", "ambient_coop")),
            ("环境二：特征文字 + 模糊图片", "Lucid（8/255, 1000步）", _fmt_old_joint("lucid"), _fmt_new("joint", "lucid")),
        ]
        for env_label, m_label, old_val, new_val in rows_spec:
            lines.append(f"| {env_label} | {m_label} | {old_val} | {new_val} |")
        lines.append("")

    lines += [
        "## 五、独立子进程复算验证",
        "",
        f"- **主评测进程 PID**：`{verification['retrieval_pid']}`；**独立复算进程 PID**：`{verification['pid']}`（严格跨进程隔离，未读取任何持久化向量缓存）。",
        f"- **全量分数比对**：重新从零编码 189 候选与全部查询，完成 `{verification['rankings_recomputed']}` 次排名、共 `{verification['score_comparisons']}` 个候选分数逐项核验，最大浮点误差 `{verification['max_score_error']:.9g}`（满足 `<= 1e-6` 容差）。",
        "- **客观性保证**：全程未根据任何检索排名重选图对、调整参数或重训。",
        "",
    ]

    report_path = run / "REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return summary


# ==============================================================================
# 流水线子进程调度器与 CLI 入口
# ==============================================================================


def run_child(run: Path, stage: str) -> None:
    if (run / "PAUSE_REQUESTED").exists():
        raise InterruptedError("本轮暂停标记存在")

    command = [
        sys.executable,
        "-X",
        "utf8",
        "-B",
        "-u",
        str(Path(__file__).resolve()),
        "--output",
        str(run),
        "--stage",
        stage,
    ]
    log = run / "logs" / f"{stage}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    status(run, "running", stage=stage, log=str(log))

    with log.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=PROJECT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        save_json(
            run / "ACTIVE_CHILD.json",
            dict(pid=process.pid, stage=stage, command=command, started_at=now()),
        )
        code = process.wait()

    if (run / "ACTIVE_CHILD.json").exists():
        try:
            (run / "ACTIVE_CHILD.json").unlink()
        except OSError:
            pass

    if code:
        if code == 2 and (run / "PAUSE_REQUESTED").exists():
            raise InterruptedError("工作进程已保存暂停断点")
        raise RuntimeError(f"阶段执行失败：{stage}；退出码 {code}；日志见 {log}")


def queue(run: Path) -> None:
    plan = validate_plan(run)

    # Stage 2: Lucid 1000 步 (epsilon=8/255) 生成（独立子进程，完成后彻底释放显存）
    m_dir = method_dir(run)
    if (m_dir / "FROZEN_IMAGE.json").is_file() and (m_dir / "generation/generation_result.json").is_file():
        validate_lucid_image(run, plan)
    else:
        run_child(run, "generate_lucid")
        validate_lucid_image(run, plan)

    freeze_images(run)

    for stage, marker in (
        ("evaluate", "RETRIEVAL_COMPLETED.json"),
        ("verify", "INDEPENDENT_VERIFICATION.json"),
        ("report", "SUMMARY.json"),
    ):
        run_child(run, stage)
        if not (run / marker).is_file():
            raise RuntimeError(f"缺少阶段完成证据文件：{marker}")

    validate_plan(run)
    inputs = read_json(run / "EVALUATION_INPUTS.json")
    freeze_json(
        run / "COMPLETED.json",
        dict(
            status="completed",
            experiment="lucid_shampoo_reselect_20261112",
            method=METHOD,
            epsilon_levels=EPSILON_LEVELS,
            steps=REQUESTED_STEPS,
            group=GROUP_KEY,
            source_id=NEW_SHAMPOO_ASIN,
            target_id=TARGET_ASIN,
            generation_result_sha256=sha256_file(method_dir(run) / "generation/generation_result.json"),
            retrieval_banks=len(inputs["bank_order"]),
            query_modes=inputs["query_modes"],
            query_rankings=inputs["expected_rankings"],
            candidate_scores=inputs["expected_score_comparisons"],
            independent_verification=True,
            plan_sha256=sha256_file(run / "EXPERIMENT_PLAN.json"),
            frozen_images_sha256=sha256_file(run / "FROZEN_IMAGES.json"),
            verification_sha256=sha256_file(run / "INDEPENDENT_VERIFICATION.json"),
            summary_sha256=sha256_file(run / "SUMMARY.json"),
            report_sha256=sha256_file(run / "REPORT.md"),
            shopping_enabled=False,
        ),
    )
    status(
        run,
        "completed",
        report=str((run / "REPORT.md").resolve()),
        query_rankings=inputs["expected_rankings"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--all", action="store_true", help="启动全流程串行流水线 (queue)")
    parser.add_argument(
        "--stage",
        choices=("queue", "generate_lucid", "evaluate", "verify", "report"),
        default="queue",
    )
    args = parser.parse_args()

    run = output_root(args.output)
    run.mkdir(parents=True, exist_ok=True)
    stage = "queue" if args.all else args.stage
    lock_file = run / ("RUN.lock" if stage == "queue" else "WORKER.lock")
    handle = lock(lock_file)

    try:
        if args.prepare_only:
            prepare(run)
            print("重选洗发水组 Lucid (8/255, 1000步) 实验计划已冻结；未读取共享记忆或正式查询。", flush=True)
            return

        if stage == "queue":
            if not (run / "EXPERIMENT_PLAN.json").exists():
                prepare(run)
            queue(run)
        elif stage == "generate_lucid":
            generate_lucid(run)
        elif stage == "evaluate":
            validate_plan(run)
            freeze_images(run)
            evaluate(run)
        elif stage == "verify":
            validate_plan(run)
            freeze_images(run)
            verify(run)
        elif stage == "report":
            validate_plan(run)
            freeze_images(run)
            report(run)
    except InterruptedError as error:
        status(run, "paused", stage=stage, message=str(error))
        raise SystemExit(2)
    except Exception as error:
        save_json(
            run / "FAILURE.json",
            dict(
                status="failed",
                stage=stage,
                error_type=type(error).__name__,
                message=str(error),
                failed_at=now(),
            ),
        )
        status(run, "failed", stage=stage, message=str(error))
        raise
    finally:
        handle.close()


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    main()
