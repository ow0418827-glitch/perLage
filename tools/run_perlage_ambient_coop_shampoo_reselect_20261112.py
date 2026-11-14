"""重选洗发水组 (07_shampoo: B09B63YYCW -> B00BDJODWI) 环境光协同 (ambient_coop) 独立生成与检索评测流水线：
在完全独立实验目录中运行：
  prepare -> generate_prefix (PerLAGE_no_pixel 四阶段前缀) -> generate_ambient_coop (ambient_coop 350步)
  -> freeze_images -> evaluate (joint + text 双查询模式、189共享池单条目替换、clean/baseline/ambient_coop 三库)
  -> verify (独立子进程从零重编码复算 22,680 个候选分数) -> report (汇总指标、视觉检验拼图、与旧洗发水组只读对照)。

严格遵守：
1. 训练与生成期严禁读取共享记忆池、proposed_pool_error_entry.json、正式查询清单及任何既有实验输出；
2. 完整生成 PerLAGE_no_pixel 前缀与 ambient_coop 成图并全部冻结后，方可在隔离银行副本中替换洗发水单条目并评测；
3. 仅评估检索排名 (shopping_enabled=False, model_api_calls_enabled=False)，不修改任何已有数据集或实验结果。
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, is_dataclass
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
from src.visual._perlage.components.colour import read_rgb, print_region_mask
from src.visual._perlage.components.core import source_ssim

BASE = PROJECT / "output/experiment_base_groups9_20261109"
RESELECT_DIR = PROJECT / "output/public_pair_selection/shampoo_reselect_20261112"
RESELECT_COMP = RESELECT_DIR / "comparison_package"
RESELECT_PKG = RESELECT_COMP / "07_shampoo"
QUERY_MULTIMODAL_RUN = PROJECT / "output/groups9_multimodal_20261110"

# 历史只读对照路径（仅在 report 阶段读取汇总，训练与检索绝不访问）
HIST_AMBIENT_RUN = PROJECT / "output/perlage_ambient_coop_groups79_20261112"
HIST_NO_PIXEL_RUN = PROJECT / "output/perlage_no_pixel_groups9_20261109"
HIST_LUCID_RUN = PROJECT / "output/lucid_groups9_20261109"
HIST_SHAMPOO_MODES = PROJECT / "output/analysis_shampoo_query_modes_20261112"
DEFAULT_RUN = PROJECT / "output/perlage_ambient_coop_shampoo_reselect_20261112"

GROUP_KEY = "07_shampoo"
PROFILE = "ambient_coop"
OLD_SHAMPOO_ASIN = "B08XYBS678"
OLD_SHAMPOO_MEMORY_ID = "groups9_reference_07_shampoo_B08XYBS678"
NEW_SHAMPOO_ASIN = "B09B63YYCW"
NEW_SHAMPOO_MEMORY_ID = "groups9_reference_07_shampoo_B09B63YYCW"
TARGET_ASIN = "B00BDJODWI"
BANKS = ("clean", "baseline", "ambient_coop")
QUERY_MODES = ("joint", "text")
STAGES = ("foreground_segmentation", "foreground_statistics", "color_harmonization", "latent_generation_refinement")


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
        "# 重选洗发水组（B09B63YYCW -> B00BDJODWI）Ambient-Coop 实验运行状态",
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
        "前缀进度见 `prefix/PerLAGE_no_pixel/07_shampoo/generation/latent_generation_refinement/`；",
        "ambient_coop 进度见 `generation/ambient_coop/07_shampoo/progress.json`；",
        "完成后将自动生成 `REPORT.md` 与 `COMPLETED.json`。",
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
    for pkg in ("src/visual/_perlage_ambient_coop", "src/visual/_perlage_no_pixel"):
        pkg_dir = PROJECT / pkg
        if pkg_dir.exists():
            paths += sorted(pkg_dir.glob("*.py"))
    paths += [
        PROJECT / p
        for p in (
            "src/visual/optimization.py",
            "src/visual/attack_completion.py",
            "src/visual/catalog_training_guard.py",
            "src/visual/retrieval_objective.py",
            "src/visual/robust_objectives.py",
            "src/visual/target_features.py",
            "src/visual/visual_retrieval.py",
            "src/visual/encoders/joint.py",
            "src/visual/encoders/clip.py",
            "src/visual/encoders/backbone.py",
            "src/visual/_perlage/initialization.py",
            "src/visual/_perlage/harmonization.py",
            "src/visual/_perlage/latent.py",
            "src/visual/_perlage/components/core.py",
            "src/visual/_perlage/components/colour.py",
            "src/visual/_perlage/components/mask.py",
            "src/visual/_perlage/components/segmentation.py",
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
    from src.visual._perlage_ambient_coop import AMBIENT_COOP_PRESETS, AmbientCoopLatentConfig
    from src.visual._perlage_no_pixel.pipeline import validate_definition

    run = output_root(run)
    run.mkdir(parents=True, exist_ok=True)

    config_path = RESELECT_PKG / "perlage_no_pixel_config.json"
    definition_path = RESELECT_PKG / "definition.json"
    features_path = RESELECT_PKG / "features.json"
    metadata_path = RESELECT_COMP / "public_metadata/source_and_target_public_metadata.json"
    source_path = (RESELECT_PKG / "source.jpg").resolve()
    target_path = (RESELECT_PKG / "target.jpg").resolve()

    definition = read_json(config_path)
    definition["data"]["source"] = str(source_path)
    definition["data"]["target"] = str(target_path)

    public_def = read_json(definition_path)
    features_doc = read_json(features_path)
    catalog_products = read_json(metadata_path)["products"]

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
        source_feature["title"] != definition["data"]["source_title"]
        or public_def["source"]["title"] != source_feature["title"]
    ):
        raise ValueError("公开源商品名不一致，禁止构造不匹配的训练键")
    if (
        source_feature.get("brand", "") != public_def["source"].get("brand", "")
        or source_feature.get("attributes") != public_def["source"].get("attributes")
    ):
        raise ValueError("公开源商品品牌或属性在 definition.json 与 features.json 间不一致")

    for evidence in source_feature["feature_evidence"]:
        field = source_catalog.get(evidence["field"], "")
        if "index" in evidence:
            field = field[evidence["index"]]
        if evidence["quote"] not in str(field):
            raise ValueError(f"源商品特点缺少公开原文证据：{evidence}")

    if definition["data"]["training_queries"] != [definition["target_feature_text"]]:
        raise ValueError("训练正文必须严格复用单一冻结目标特点句子")
    if (
        definition["target_feature_text"] != target_feature["canonical_sentence"]
        or definition["target_feature_text"] != public_def["target_feature_text"]
    ):
        raise ValueError("目标特点句子与公开选中包不一致")

    # 在 CPU 侧提前验证两个方法的公开配置合法性
    validate_definition(copy.deepcopy(definition))
    AmbientCoopLatentConfig(**AMBIENT_COOP_PRESETS[PROFILE]).validate()

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
            config_path,
            definition_path,
            features_path,
            metadata_path,
            source_path,
            target_path,
        )
    }

    profile_config = (
        asdict(AMBIENT_COOP_PRESETS[PROFILE])
        if is_dataclass(AMBIENT_COOP_PRESETS[PROFILE])
        else copy.deepcopy(AMBIENT_COOP_PRESETS[PROFILE])
    )

    plan = dict(
        schema_version=1,
        experiment="perlage_ambient_coop_shampoo_reselect_20261112",
        group=GROUP_KEY,
        groups=[GROUP_KEY],
        profiles=[PROFILE],
        profile_configs={PROFILE: profile_config},
        source_id=source_id,
        target_id=target_id,
        replaces_memory_id=OLD_SHAMPOO_MEMORY_ID,
        new_memory_id=NEW_SHAMPOO_MEMORY_ID,
        generation_inputs={
            GROUP_KEY: dict(
                definition=definition,
                public_definition=public_def,
                source_public_record=source_public_record,
                prefix_method_root=str((run / "prefix/PerLAGE_no_pixel" / GROUP_KEY).resolve()),
                prefix_root=str((run / "prefix/PerLAGE_no_pixel" / GROUP_KEY / "generation").resolve()),
                ambient_coop_root=str((run / "generation" / PROFILE / GROUP_KEY).resolve()),
                files=bindings,
            )
        },
        implementation=implementation_files(),
        generation_seed=profile_config.get("seed", 42),
        training="original_frozen_single_target_feature_sentence",
        training_key="own_public_title_brand_attributes_feature_text_reconstructed_without_pool",
        prefix_policy="fresh_four_stage_perlage_no_pixel_on_reselected_source_no_pixel_optimization",
        query_modes=list(QUERY_MODES),
        query_weights={"joint": 0.5, "text": 0.0},
        key_image_weight=0.5,
        index_text="product_feature",
        shared_pool_count=189,
        pool_replacement_scope="single_group_07_shampoo_B08XYBS678_replaced_with_B09B63YYCW_in_isolated_copy",
        all_images_frozen_before_evaluation=True,
        pixel_optimization=False,
        shopping_enabled=False,
        model_api_calls_enabled=False,
        formal_rankings_used_for_parameter_selection=False,
        training_reads_shared_pool=False,
        caveats=[
            "本轮使用独立隔离重选的真人类洗发水原商品 B09B63YYCW（BELLISSO Biotin Shampoo）替代原误选的洗手液 B08XYBS678（Dove Hand Wash），目标商品保持 Aveda B00BDJODWI 不变。",
            "因 B09B63YYCW 尚未生成过 PerLAGE_no_pixel 成图，本流水线先在隔离目录从头完整运行四阶段 PerLAGE_no_pixel（60 步潜变量细化，零像素扰动）作为锚点前缀，再运行 ambient_coop（350 步，主体 6 级微光影＋背景 28 级环境光协同，ROI 印刷区零改动）。",
            "评测阶段在复制的 189 条共享长期记忆池隔离副本中，仅替换 07_shampoo 原条目为 B09B63YYCW 模拟参考条目（ObservedPurchase=false, RecordKind=simulated_gallery），其余 188 条及原有数据文件保持严格只读不变。",
            "同时评测 multimodal joint（图文各半，复用 20261110 独立模糊照片与 20 条特点句）与 text（纯文字，复用 20261109 同 20 条特点句）；仅统计检索排名，不执行购物。",
            "训练代理与指定评估编码器均为 ViT-B-32 (openai)，属于同编码器评估而非独立留出黑盒迁移；正式 20 条特点问句中有 1 条与唯一训练正文精确重合。",
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
    if (
        plan.get("group") != GROUP_KEY
        or plan.get("groups") != [GROUP_KEY]
        or plan.get("profiles") != [PROFILE]
        or plan.get("shopping_enabled") is not False
    ):
        raise ValueError("实验计划范围、档位或停止购物标记发生篡改")
    for path, expected in plan["implementation"].items():
        _check(path, expected)
    for item in plan["generation_inputs"].values():
        for path, expected in item["files"].items():
            _check(path, expected)
    return plan


# ==============================================================================
# Stage 2: generate_prefix (PerLAGE_no_pixel 四阶段前缀) & Stage 3: generate_ambient_coop
# ==============================================================================


def prefix_method_dir(run: Path) -> Path:
    return run / "prefix/PerLAGE_no_pixel" / GROUP_KEY


def ambient_dir(run: Path) -> Path:
    return run / "generation" / PROFILE / GROUP_KEY


def validate_prefix_image(run: Path, plan: dict | None = None) -> dict:
    from src.visual.attack_completion import validate_attack_completion
    from src.visual._perlage_no_pixel.evidence import validate_complete

    plan = plan or read_json(run / "EXPERIMENT_PLAN.json")
    item = plan["generation_inputs"][GROUP_KEY]
    definition = item["definition"]
    method_root = prefix_method_dir(run)
    gen_dir = method_root / "generation"

    ready = read_json(gen_dir / "ready.json")
    completion = read_json(gen_dir / "completion.json")
    frozen = read_json(method_root / "FROZEN_IMAGE.json")
    frozen_png = method_root / "frozen.png"

    validate_complete(ready, gen_dir / "adversarial.png")
    validate_attack_completion(
        completion,
        requested_steps=60,
        expected={
            "method": "PerLAGE_no_pixel",
            "source_image_sha256": sha256_file(definition["data"]["source"]),
            "target_image_sha256": sha256_file(definition["data"]["target"]),
        },
    )
    if frozen.get("status") != "frozen" or frozen.get("image_sha256") != sha256_file(frozen_png):
        raise ValueError("PerLAGE_no_pixel 前缀冻结图摘要不符")
    if sha256_file(frozen_png) != sha256_file(gen_dir / "adversarial.png"):
        raise ValueError("PerLAGE_no_pixel 冻结副本与原始生成图不一致")
    if not np.array_equal(read_rgb(gen_dir / "adversarial.png"), read_rgb(gen_dir / "generative_anchor.png")):
        raise ValueError("PerLAGE_no_pixel 存在非零像素残差")
    return dict(
        image=str(frozen_png.resolve()),
        raw_adversarial=str((gen_dir / "adversarial.png").resolve()),
        sha256=sha256_file(frozen_png),
        ready=str((gen_dir / "ready.json").resolve()),
        ready_sha256=sha256_file(gen_dir / "ready.json"),
        frozen_json=str((method_root / "FROZEN_IMAGE.json").resolve()),
        frozen_json_sha256=sha256_file(method_root / "FROZEN_IMAGE.json"),
    )


def validate_ambient_image(run: Path, plan: dict | None = None) -> dict:
    from src.visual._perlage_ambient_coop import validate_ambient_coop_complete

    plan = plan or read_json(run / "EXPERIMENT_PLAN.json")
    item = plan["generation_inputs"][GROUP_KEY]
    definition = item["definition"]
    config = plan["profile_configs"][PROFILE]
    root = ambient_dir(run)
    ready = read_json(root / "ready.json")
    validate_ambient_coop_complete(ready, root / "adversarial.png")

    if (
        ready.get("profile") != PROFILE
        or ready.get("latent_config") != config
        or ready["stages"]["latent_generation_refinement"]["steps_requested"] != config["steps"]
    ):
        raise ValueError("ambient_coop 完成图片档位或细化步数不符")

    for role in ("source", "target"):
        if ready.get(role + "_sha256") != sha256_file(definition["data"][role]):
            raise ValueError(f"ambient_coop 绑定了错误的公开输入：{role}")

    identity = read_json(ready["method_identity"])
    if identity["public_definition"] != definition:
        raise ValueError("ambient_coop 公开定义与计划不一致")
    if identity["source_public_record"] != item["source_public_record"]:
        raise ValueError("ambient_coop 公开源商品特征键与计划不一致")
    if Path(identity["prefix_binding"]["root"]).resolve() != Path(item["prefix_root"]).resolve():
        raise ValueError("ambient_coop 未绑定本轮隔离生成的 PerLAGE_no_pixel 前缀")

    return dict(
        image=str((root / "adversarial.png").resolve()),
        sha256=sha256_file(root / "adversarial.png"),
        ready=str((root / "ready.json").resolve()),
        ready_sha256=sha256_file(root / "ready.json"),
    )


def generate_prefix(run: Path) -> None:
    from src.visual.attack_completion import validate_attack_completion
    from src.visual.catalog_training_guard import catalog_training_guard
    from src.visual.optimization import generate_attack

    plan = validate_plan(run)
    if (run / "PAUSE_REQUESTED").exists():
        raise InterruptedError("本轮已请求暂停")

    item = plan["generation_inputs"][GROUP_KEY]
    definition = copy.deepcopy(item["definition"])
    method_root = prefix_method_dir(run)
    gen_dir = method_root / "generation"
    frozen_json_path = method_root / "FROZEN_IMAGE.json"
    frozen_png_path = method_root / "frozen.png"

    if frozen_json_path.is_file() and frozen_png_path.is_file() and (gen_dir / "ready.json").is_file():
        validate_prefix_image(run, plan)
        print(json.dumps(dict(stage="prefix_already_completed", group=GROUP_KEY), ensure_ascii=False), flush=True)
        return

    method_root.mkdir(parents=True, exist_ok=True)
    gen_dir.mkdir(parents=True, exist_ok=True)

    source_img = Path(definition["data"]["source"]).resolve()
    target_img = Path(definition["data"]["target"]).resolve()
    pub_def = item["public_definition"]
    source_record = copy.deepcopy(pub_def["source"])
    source_record["image"] = str(source_img)
    source_record["image_sha256"] = sha256_file(source_img)
    target_record = copy.deepcopy(pub_def["target"])
    target_record["image"] = str(target_img)
    target_record["image_sha256"] = sha256_file(target_img)

    public_binding = dict(
        method="PerLAGE_no_pixel",
        group=GROUP_KEY,
        selection_sha256=sha256_file(RESELECT_PKG / "definition.json"),
        source=source_record,
        target=target_record,
        shared_pool_scope="isolated_prefix_generation_no_pool_read",
        threat_model="black_box_provisional",
    )
    freeze_json(method_root / "PUBLIC_RUN.json", public_binding)

    tick = time.perf_counter()
    with catalog_training_guard(
        allowed_output_roots=[run, RESELECT_PKG],
        private_paths=_private_paths_for_training(run),
    ) as audit:
        _, metadata = generate_attack(
            source=source_img,
            target=target_img,
            output_dir=gen_dir,
            method="PerLAGE_no_pixel",
            perlage_no_pixel_config=definition,
            device="cuda",
        )
        freeze_json(gen_dir / "completion.json", metadata)

    elapsed = time.perf_counter() - tick
    freeze_json(
        method_root / "TRAINING_ACCESS_AUDIT.json",
        {k: v for k, v in audit.items() if k != "active"},
    )
    if audit.get("blocked_accesses"):
        raise PermissionError(f"训练访问守护检测到越权访问：{audit['blocked_accesses']}")

    adversarial_png = gen_dir / "adversarial.png"
    anchor_png = gen_dir / "generative_anchor.png"
    if not np.array_equal(read_rgb(adversarial_png), read_rgb(anchor_png)):
        raise AssertionError("PerLAGE_no_pixel 存在非零像素级残差")

    shutil.copyfile(adversarial_png, frozen_png_path)
    shutil.copyfile(adversarial_png, method_root / "adversarial.png")
    shutil.copyfile(anchor_png, method_root / "generative_anchor.png")
    shutil.copyfile(gen_dir / "ready.json", method_root / "ready.json")
    shutil.copyfile(gen_dir / "completion.json", method_root / "completion.json")
    freeze_json(method_root / "definition.json", definition)

    frozen_hash = sha256_file(frozen_png_path)
    frozen_data = dict(
        schema_version=1,
        method="PerLAGE_no_pixel",
        status="frozen",
        image=str(frozen_png_path.resolve()),
        image_sha256=frozen_hash,
        source=source_record,
        target=target_record,
        metadata=metadata,
        requested_steps=60,
        observed_seconds=elapsed,
        public_binding=public_binding,
        public_binding_sha256=digest(public_binding),
    )
    freeze_json(frozen_json_path, frozen_data)

    validate_attack_completion(
        metadata,
        requested_steps=60,
        expected={
            "method": "PerLAGE_no_pixel",
            "source_image_sha256": source_record["image_sha256"],
            "target_image_sha256": target_record["image_sha256"],
        },
    )
    validate_prefix_image(run, plan)
    print(
        json.dumps(
            dict(stage="prefix_generation_completed", group=GROUP_KEY, sha256=frozen_hash, seconds=round(elapsed, 2)),
            ensure_ascii=False,
        ),
        flush=True,
    )


def generate_ambient(run: Path) -> None:
    from src.visual.catalog_training_guard import catalog_training_guard
    from src.visual._perlage_ambient_coop import generate_ambient_coop

    plan = validate_plan(run)
    if (run / "PAUSE_REQUESTED").exists():
        raise InterruptedError("本轮已请求暂停")

    validate_prefix_image(run, plan)
    item = plan["generation_inputs"][GROUP_KEY]
    definition = copy.deepcopy(item["definition"])
    root = ambient_dir(run)

    with catalog_training_guard(
        allowed_output_roots=[run, RESELECT_PKG],
        private_paths=_private_paths_for_training(run),
    ) as audit:
        image, ready = generate_ambient_coop(
            definition["data"]["source"],
            definition["data"]["target"],
            root,
            definition=definition,
            prefix_root=item["prefix_root"],
            profile=PROFILE,
            source_public_record=item["source_public_record"],
            device="cuda",
            pause_path=run / "PAUSE_REQUESTED",
        )

    freeze_json(
        root / "TRAINING_ACCESS_AUDIT.json",
        {k: v for k, v in audit.items() if k != "active"},
    )
    if audit.get("blocked_accesses"):
        raise PermissionError(f"ambient_coop 访问守护检测到越权访问：{audit['blocked_accesses']}")
    if not image or ready.get("status") != "completed":
        raise InterruptedError("已保存 ambient_coop 断点")

    validate_ambient_image(run, plan)
    print(
        json.dumps(dict(stage="ambient_coop_generation_completed", profile=PROFILE, group=GROUP_KEY), ensure_ascii=False),
        flush=True,
    )


def freeze_images(run: Path) -> dict:
    plan = validate_plan(run)
    prefix_asset = validate_prefix_image(run, plan)
    amb_asset = validate_ambient_image(run, plan)
    payload = dict(
        status="frozen",
        images={
            "baseline": {GROUP_KEY: prefix_asset},
            PROFILE: {GROUP_KEY: amb_asset},
        },
        plan_sha256=sha256_file(run / "EXPERIMENT_PLAN.json"),
        formal_rankings_used=False,
    )
    freeze_json(run / "FROZEN_IMAGES.json", payload)
    return payload


# ==============================================================================
# Stage 4-6: Isolated Pool Replacement, Evaluation (joint + text), Verification, Report
# ==============================================================================


def _frozen_generation(run: Path) -> dict:
    run = Path(run).resolve()
    plan_file = run / "EXPERIMENT_PLAN.json"
    plan = read_json(plan_file)
    if plan.get("groups") != [GROUP_KEY] or plan.get("profiles") != [PROFILE] or plan.get("shopping_enabled") is not False:
        raise ValueError("实验计划范围无效")

    frozen_file = run / "FROZEN_IMAGES.json"
    frozen = read_json(frozen_file)
    if frozen.get("status") != "frozen" or set(frozen.get("images", {})) != {"baseline", PROFILE}:
        raise ValueError("必须同时冻结 baseline 前缀成图与 ambient_coop 最终成图后方可评测")
    if frozen.get("plan_sha256") != sha256_file(plan_file):
        raise ValueError("冻结图片登记未绑定当前实验计划")

    validate_prefix_image(run, plan)
    validate_ambient_image(run, plan)
    _check(frozen["images"]["baseline"][GROUP_KEY]["image"], frozen["images"]["baseline"][GROUP_KEY]["sha256"])
    _check(frozen["images"][PROFILE][GROUP_KEY]["image"], frozen["images"][PROFILE][GROUP_KEY]["sha256"])
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
    if replacement_entry.get("ObservedPurchase") is not False or replacement_entry.get("Completed") is not False or replacement_entry.get("Actions") != []:
        raise ValueError("未观察到购买的参考条目不得伪造 Completed 或 Actions 购买轨迹")

    # 检查其余 188 条与新条目零碰撞，且无历史替换残留
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

    expected_modified = ("baseline", PROFILE)
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

    # 校验隔离池新条目特征键与 ambient_coop 训练公开全身键完全一致
    source_title = gen_item["definition"]["data"]["source_title"]
    trained_key = build_source_public_key(gen_item["source_public_record"], source_title)
    indexed_key = memory_index_text(replacement_entry, "product_feature")
    if indexed_key != trained_key:
        raise ValueError("新洗发水共享池特征检索键与 ambient_coop 训练公开键不一致")

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

    baseline_asset = frozen["images"]["baseline"][GROUP_KEY]
    amb_asset = frozen["images"][PROFILE][GROUP_KEY]
    for asset in (baseline_asset, amb_asset):
        bindings[asset["image"]] = asset["sha256"]
        bindings[asset["ready"]] = asset["ready_sha256"]

    banks = build_isolated_banks(
        entries,
        pool_path,
        replacement_entry,
        image_assets={
            "baseline": {"image": baseline_asset["image"], "sha256": baseline_asset["sha256"]},
            PROFILE: {"image": amb_asset["image"], "sha256": amb_asset["sha256"]},
        },
        resolve_paths=True,
    )

    bank_paths = {}
    for name in BANKS:
        path = run / "banks" / f"{name}.json"
        freeze_json(path, banks[name])
        bank_paths[name] = dict(path=str(path.resolve()), sha256=sha256_file(path))
        bindings[str(path.resolve())] = sha256_file(path)

    # 加载正式查询集（joint + text），确保 20 条特点正文一致
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
    """两张完整图片冻结后，新建编码器执行 joint 与 text 双模式全库 189 条零磁盘缓存检索评测。"""
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
    """严格在独立子进程从零重编码全部 189 条候选与查询，逐项比对全部 22,680 个分数。"""
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


def _compute_baseline_visual(run: Path, plan: dict) -> dict:
    item = plan["generation_inputs"][GROUP_KEY]
    source_path = item["definition"]["data"]["source"]
    gen_dir = Path(item["prefix_root"])
    meta = read_json(gen_dir / "latent_generation_refinement/metadata.json")
    source_rgb = read_rgb(source_path)
    final_rgb = read_rgb(gen_dir / "adversarial.png")
    fg_mask = read_rgb(gen_dir / "foreground_segmentation/source/foreground.png")[..., 0] > 0
    roi_mask = print_region_mask(source_rgb.shape, item["definition"]["print_regions"])
    diff = np.abs(final_rgb.astype(np.float32) - source_rgb.astype(np.float32))
    t_final = torch.as_tensor(final_rgb.transpose(2, 0, 1), dtype=torch.float32)[None] / 255.0
    t_source = torch.as_tensor(source_rgb.transpose(2, 0, 1), dtype=torch.float32)[None] / 255.0
    best = meta.get("best", {})
    history = meta.get("history", [])
    steps = max((r["step"] for r in history), default=meta.get("config", {}).get("steps", 60))
    return dict(
        steps=int(steps),
        best_step=int(best.get("step", steps)),
        accepted=int(meta.get("accepted_updates", 0)),
        rejected=int(meta.get("rejected_updates", 0)),
        source_ssim=float(source_ssim(t_final, t_source)),
        source_mae_levels=float(diff.mean()),
        fg_source_mae_levels=float(diff[fg_mask].mean()),
        bg_source_mae_levels=float(diff[~fg_mask].mean()),
        roi_source_mae_levels=float(diff[roi_mask].mean()) if roi_mask.any() else 0.0,
        fg_anchor_mae_levels=0.0,
        bg_anchor_mae_levels=0.0,
        fg_latent_max_levels=0,
        fg_latent_mae_levels=0.0,
        bg_latent_max_levels=0,
        bg_latent_mae_levels=0.0,
        roi_anchor_max_levels=0,
        bg_mean_rgb_shift_levels=[0.0, 0.0, 0.0],
        halo_error=float(best.get("halo_error", 0.0)),
        internal_selection_score=float(best.get("selection", [0.0])[-1]) if isinstance(best.get("selection"), list) else 0.0,
    )


def _compute_ambient_visual(run: Path, plan: dict) -> dict:
    item = plan["generation_inputs"][GROUP_KEY]
    gen_dir = ambient_dir(run)
    prefix_dir = Path(item["prefix_root"])
    meta = read_json(gen_dir / "metadata.json")
    fc = meta["final_check"]
    final_rgb = read_rgb(gen_dir / "adversarial.png")
    anchor_rgb = read_rgb(prefix_dir / "adversarial.png")
    roi_mask = print_region_mask(anchor_rgb.shape, item["definition"]["print_regions"])
    roi_change = np.abs(final_rgb.astype(np.int16) - anchor_rgb.astype(np.int16))[roi_mask] if roi_mask.any() else np.array([0])
    return dict(
        steps=int(meta["steps_completed"]),
        best_step=int(meta["best"]["step"]),
        accepted=int(meta["accepted_updates"]),
        rejected=int(meta["rejected_updates"]),
        source_ssim=float(fc["source_ssim"]),
        source_mae_levels=float(fc["source_mae_levels"]),
        fg_source_mae_levels=float(fc["fg_source_mae_levels"]),
        bg_source_mae_levels=float(fc["bg_source_mae_levels"]),
        fg_anchor_mae_levels=float(fc["fg_anchor_mae_levels"]),
        bg_anchor_mae_levels=float(fc["bg_anchor_mae_levels"]),
        fg_latent_max_levels=int(meta["fg_latent_max_levels"]),
        fg_latent_mae_levels=float(meta["fg_latent_mae_levels"]),
        bg_latent_max_levels=int(meta["bg_latent_max_levels"]),
        bg_latent_mae_levels=float(meta["bg_latent_mae_levels"]),
        roi_anchor_max_levels=int(roi_change.max()),
        bg_mean_rgb_shift_levels=[float(x) for x in fc.get("bg_mean_rgb_shift_levels", [0.0, 0.0, 0.0])],
        ambient_global_levels=[float(x) for x in fc.get("ambient_global_levels", [0.0, 0.0, 0.0])],
        halo_error=float(fc["halo_error"]),
        joint_score=float(fc["joint_scores"][0][0]),
    )


def _save_visual_contact_sheet(run: Path, plan: dict) -> Path:
    item = plan["generation_inputs"][GROUP_KEY]
    src = Image.open(item["definition"]["data"]["source"]).convert("RGB")
    base = Image.open(prefix_method_dir(run) / "frozen.png").convert("RGB")
    amb = Image.open(ambient_dir(run) / "adversarial.png").convert("RGB")

    # 5x 放大绝对差异图（relative to source and relative to baseline anchor）
    arr_src = np.asarray(src, dtype=np.int16)
    arr_base = np.asarray(base, dtype=np.int16)
    arr_amb = np.asarray(amb, dtype=np.int16)
    diff_anchor_x5 = Image.fromarray(np.clip(np.abs(arr_amb - arr_base) * 5, 0, 255).astype(np.uint8))

    w, h = src.size
    pad = 16
    banner = 36
    panels = [
        ("1. Reselected Clean (B09B63YYCW)", src),
        ("2. PerLAGE_no_pixel Prefix", base),
        ("3. Ambient-Coop Final (350s)", amb),
        ("4. |Ambient - Prefix| x5", diff_anchor_x5),
    ]
    canvas = Image.new("RGB", (len(panels) * w + (len(panels) + 1) * pad, h + banner + pad), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    for idx, (label, img) in enumerate(panels):
        x = pad + idx * (w + pad)
        draw.text((x, 10), label, fill=(20, 20, 20))
        canvas.paste(img, (x, banner))
    out = run / "comparison_visual_07_shampoo.png"
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


def _load_historical_reference() -> dict:
    """只读读取历史旧洗发水 (B08XYBS678 Dove 洗手液) 指标供对比展示。"""
    ref = {}
    if (HIST_AMBIENT_RUN / "SUMMARY.json").is_file():
        s = read_json(HIST_AMBIENT_RUN / "SUMMARY.json")
        ref["joint_b08"] = {
            "note": "历史 07_shampoo (B08XYBS678 Dove 洗手液)，与 09_toothbrush 同步双图替换在 189 共享池中评测（含 B08XYBS678 条目文字，非同商品）",
            "banks": {b: s["banks"][b]["metrics"]["07_shampoo"] for b in ("clean", "baseline", "lucid", "ambient_coop") if b in s.get("banks", {})},
            "visual": {k: s["visual_metrics"][k]["07_shampoo"] for k in ("baseline", "ambient_coop") if k in s.get("visual_metrics", {})},
        }
    if (HIST_SHAMPOO_MODES / "AUDIT.json").is_file():
        a = read_json(HIST_SHAMPOO_MODES / "AUDIT.json")
        np_text = a.get("results", {}).get("perlage_no_pixel", {}).get("A_feature_text", {})
        if np_text:
            ref["text_b08"] = {
                "note": "历史 07_shampoo (B08XYBS678 Dove 洗手液) 纯文字特征检索汇总",
                "clean": np_text.get("clean", {}).get("stats"),
                "baseline": np_text.get("modified", {}).get("stats"),
            }
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
    visual = {
        "baseline": _compute_baseline_visual(run, plan),
        PROFILE: _compute_ambient_visual(run, plan),
    }
    contact_sheet = _save_visual_contact_sheet(run, plan)

    modes_summary = {}
    for mode in QUERY_MODES:
        bank_entries = {}
        for bank in BANKS:
            rows = results[mode][bank]["rankings"]
            comparisons = {}
            for ref in [b for b in BANKS if b != bank]:
                prev_rows = results[mode][ref]["rankings"]
                before = results[mode][ref]["metrics"]
                after = results[mode][bank]["metrics"]
                comparisons[ref] = dict(
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

    historical = _load_historical_reference()
    summary = dict(
        schema_version=1,
        status="completed",
        scope="retrieval_only",
        group=GROUP_KEY,
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
        historical_b08xybs678_reference=historical,
    )
    freeze_json(run / "SUMMARY.json", summary)

    bank_labels = {
        "clean": "原图库 (clean, B09B63YYCW 原图)",
        "baseline": "无像素感知前缀 (PerLAGE_no_pixel 基线)",
        "ambient_coop": "环境光背景协同＋主体微光影 (ambient_coop)",
    }
    mode_labels = {
        "joint": "主评测口径：特点文字＋独立模糊照片 (joint, 图文各半)",
        "text": "对照口径：特点纯文字 (text, 无查询图片)",
    }

    lines = [
        "# 重选洗发水组（B09B63YYCW -> B00BDJODWI）Ambient-Coop 生成与共享记忆检索报告",
        "",
        "## 一、实验范围与隔离约束核验",
        "",
        f"- **重选原商品**：`{inputs['source_id']}`（BELLISSO Biotin Shampoo，真人类洗发水，DINOv2 灰度外形分 `0.6974 >= 0.6011`），在 189 共享池隔离副本中替换原洗手液条目 `{inputs['replaced_memory_id']}`（其余 188 条及只读基础库完全不变）。",
        f"- **固定目标商品**：`{inputs['target_id']}`（Aveda Scalp Benefits Balancing Shampoo 33.8 OZ），唯一训练正文：`{plan['generation_inputs'][GROUP_KEY]['definition']['target_feature_text']}`。",
        f"- **参考标记与购买约束**：新条目保留 `ObservedPurchase=false` 与 `RecordKind=simulated_gallery`（{summary['record_kinds']['purchase_memory']} 条真实购买记忆 + {summary['record_kinds']['simulated_gallery']} 条模拟参考）；`shopping_executed={summary['shopping_executed']}`（未执行任何购物与大模型对话调用）。",
        f"- **编码器口径**：训练代理与指定检索编码器均为 `ViT-B-32 (openai)`，属同编码器公开权重评测。",
        "",
        "## 二、共享长期记忆库检索排名结果（189 条完整库，20 条正式请求）",
        "",
    ]

    for mode in QUERY_MODES:
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
                f"{m['top1']}/20（{m['top1_rate']:.1%}） | {m['top3']}/20（{m['top3_rate']:.1%}） | "
                f"{m['top5']}/20（{m['top5_rate']:.1%}） | {m['mean_rank']:.3f} | "
                f"{m['min_rank']}–{m['max_rank']} | {m['mrr']:.5f} |"
            )
        lines += [
            "",
            f"**{mode} 模式配对增量（Ambient-Coop vs 同源对照基准）**：",
            "",
            "| 对照基准 | Top-1 净增 | Top-3 净增 | Top-5 净增 | 配对改善/退化/持平 | 平均名次改善↓ | MRR 增量↑ |",
            "|---|---:|---:|---:|:---:|---:|---:|",
        ]
        for ref in ("clean", "baseline"):
            d = modes_summary[mode]["ambient_coop"]["comparisons"][ref]
            lines.append(
                f"| {bank_labels[ref]} | {d['top1_gain']:+d} | {d['top3_gain']:+d} | {d['top5_gain']:+d} | "
                f"{d['improved_ranks']} / {d['worsened_ranks']} / {d['tied_ranks']} | "
                f"{d['mean_rank_reduction']:+.3f} | {d['mrr_gain']:+.5f} |"
            )
        lines.append("")

    lines += [
        "## 三、外观自然度与分区约束统计",
        "",
        "| 方案 | 步数（最佳步） | 接受/拒绝 | SSIM↑ | 全图 MAE (级)↓ | 主体相对原图 MAE | 背景相对原图 MAE | 主体相对基线 Max/MAE | 背景相对基线 Max/MAE | ROI 印刷区 Max 改动 | 背景平均 RGB 偏移 (级) |",
        "|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for key, label in (("baseline", bank_labels["baseline"]), (PROFILE, bank_labels["ambient_coop"])):
        v = visual[key]
        rgb = v["bg_mean_rgb_shift_levels"]
        lines.append(
            f"| {label} | {v['steps']}（{v['best_step']}） | {v['accepted']}/{v['rejected']} | "
            f"{v['source_ssim']:.4f} | {v['source_mae_levels']:.2f} | "
            f"{v['fg_source_mae_levels']:.2f} | {v['bg_source_mae_levels']:.2f} | "
            f"{v['fg_latent_max_levels']} / {v['fg_latent_mae_levels']:.2f} | "
            f"{v['bg_latent_max_levels']} / {v['bg_latent_mae_levels']:.2f} | "
            f"{v['roi_anchor_max_levels']} | ({rgb[0]:+.1f}, {rgb[1]:+.1f}, {rgb[2]:+.1f}) |"
        )
    lines += [
        "",
        f"视觉拼接对比图已保存至 `{contact_sheet.name}`。",
        "",
    ]

    # 首条请求前五名竞争详情（仅保留 joint 主模式）
    lines += [
        "## 四、首条正式图文请求 (feature_01) 前五名竞争情况（joint 模式）",
        "",
    ]
    for bank in BANKS:
        lines += [
            f"- **{bank_labels[bank]}**（源 `{inputs['source_id']}` 排第 `{modes_summary['joint'][bank]['per_query_source'][0]['rank']}`，得分 `{modes_summary['joint'][bank]['per_query_source'][0]['score']:.5f}`）：",
        ]
        for c in modes_summary["joint"][bank]["query01_top5"]:
            tag = " **[本轮重选源商品]**" if c["is_reselected_source"] else ""
            lines.append(f"  - #{c['rank']}: `{c['asin']}` ({c['brand']}) — {c['title'][:70]} (`{c['score']:.5f}`){tag}")
    lines.append("")

    if historical:
        lines += [
            "## 五、与旧洗发水组 (B08XYBS678 Dove 洗手液) 历史结果对照参考",
            "",
            "> **严谨声明**：历史旧组原商品为 `B08XYBS678`（Dove 泡沫洗手液），且 `perlage_ambient_coop_groups79_20261112` 中与 `09_toothbrush` 同步双图替换评测；本轮重选原商品为 `B09B63YYCW`（Bellisso Biotin Shampoo），仅替换洗发水单条目。下表并列展示两套原商品在各自 189 共享池与相同 20 条洗发水问句下的实际检索表现：",
            "",
        ]
        if "joint_b08" in historical or "text_b08" in historical:
            lines += [
                "| 查询口径 | 源商品组别 | 图库 / 方案 | Top-1 | Top-3 | Top-5 | 平均名次↓ | MRR↑ |",
                "|---|---|---|---:|---:|---:|---:|---:|",
            ]
        if "joint_b08" in historical:
            jb = historical["joint_b08"]["banks"]
            for b, b_label in (("clean", "原图库 (clean)"), ("baseline", "PerLAGE_no_pixel"), ("ambient_coop", "ambient_coop")):
                if b in jb:
                    mh = jb[b]
                    lines.append(
                        f"| joint (图文各半) | 旧组 B08XYBS678 (洗手液) | {b_label} | {mh['top1']}/20 | {mh['top3']}/20 | {mh['top5']}/20 | {mh['mean_rank']:.3f} | {mh['mrr']:.5f} |"
                    )
                mc = modes_summary["joint"][b]["metrics"]
                lines.append(
                    f"| joint (图文各半) | **新选 B09B63YYCW (真洗发水)** | {b_label} | {mc['top1']}/20 | {mc['top3']}/20 | {mc['top5']}/20 | {mc['mean_rank']:.3f} | {mc['mrr']:.5f} |"
                )
        if "text_b08" in historical:
            tb = historical["text_b08"]
            for b, hist_key in (("clean", "clean"), ("baseline", "baseline")):
                if tb.get(hist_key):
                    st = tb[hist_key]
                    lines.append(
                        f"| text (纯文字) | 旧组 B08XYBS678 (洗手液) | {b} | {st['top1']}/20 | {st['top3']}/20 | {st['top5']}/20 | {st['mean']:.3f} | — |"
                    )
                mc = modes_summary["text"][b]["metrics"]
                lines.append(
                    f"| text (纯文字) | **新选 B09B63YYCW (真洗发水)** | {b} | {mc['top1']}/20 | {mc['top3']}/20 | {mc['top5']}/20 | {mc['mean_rank']:.3f} | {mc['mrr']:.5f} |"
                )
            mc_amb = modes_summary["text"]["ambient_coop"]["metrics"]
            lines.append(
                f"| text (纯文字) | **新选 B09B63YYCW (真洗发水)** | ambient_coop | {mc_amb['top1']}/20 | {mc_amb['top3']}/20 | {mc_amb['top5']}/20 | {mc_amb['mean_rank']:.3f} | {mc_amb['mrr']:.5f} |"
            )
        lines.append("")

    lines += [
        "## 六、独立子进程复算验证",
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

    # Stage 2: PerLAGE_no_pixel 四阶段前缀（独立子进程，用完彻底释放显存）
    p_dir = prefix_method_dir(run)
    if (p_dir / "FROZEN_IMAGE.json").is_file() and (p_dir / "generation/ready.json").is_file():
        validate_prefix_image(run, plan)
    else:
        run_child(run, "generate_prefix")
        validate_prefix_image(run, plan)

    # Stage 3: Ambient-Coop 350 步生成（独立子进程）
    a_dir = ambient_dir(run)
    if (a_dir / "ready.json").is_file():
        validate_ambient_image(run, plan)
    else:
        run_child(run, "generate_ambient_coop")
        validate_ambient_image(run, plan)

    # 冻结两张完整图片后再开放评测与独立复算
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
            experiment="perlage_ambient_coop_shampoo_reselect_20261112",
            group=GROUP_KEY,
            source_id=NEW_SHAMPOO_ASIN,
            target_id=TARGET_ASIN,
            prefix_ready_sha256=sha256_file(prefix_method_dir(run) / "generation/ready.json"),
            ambient_ready_sha256=sha256_file(ambient_dir(run) / "ready.json"),
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
        choices=("queue", "generate_prefix", "generate_ambient_coop", "evaluate", "verify", "report"),
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
            print("重选洗发水组实验计划已冻结；未读取共享记忆或正式查询。", flush=True)
            return

        if stage == "queue":
            if not (run / "EXPERIMENT_PLAN.json").exists():
                prepare(run)
            queue(run)
        elif stage == "generate_prefix":
            generate_prefix(run)
        elif stage == "generate_ambient_coop":
            generate_ambient(run)
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
