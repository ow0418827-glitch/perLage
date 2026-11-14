"""第七组洗发水与第九组牙刷“环境光背景协同调整＋主体微光影优化”（Ambient-Coop）双组实验调度器：
支持无人值守流水线编排：prepare -> generate (07_shampoo, 09_toothbrush) -> freeze_images -> evaluate -> verify -> report。
集成文件排他锁、访问安全守门员与严格的子进程解耦调用。
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, is_dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
os.environ.update(
    HF_HUB_OFFLINE="1",
    TRANSFORMERS_OFFLINE="1",
    TOKENIZERS_PARALLELISM="false",
    PYTHONDONTWRITEBYTECODE="1",
    PYTHONUTF8="1"
)

from src.visual.audit import freeze_json, read_json, save_json
from src.visual.image_utils import sha256_file

BASE = PROJECT / "output/experiment_base_groups9_20261109"
PREVIOUS_BASELINE = PROJECT / "output/perlage_no_pixel_groups9_20261109"
SHOPPING = PROJECT / "output/groups9_multimodal_20261110"
DEFAULT_RUN = PROJECT / "output/perlage_ambient_coop_groups79_20261112"

GROUPS = ("07_shampoo", "09_toothbrush")
PROFILES = ("ambient_coop",)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def check(path: str | Path, expected: str) -> None:
    if sha256_file(path) != expected:
        raise ValueError(f"冻结文件摘要变化：{path}")


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
        raise RuntimeError("已有本轮队列或工作进程正在执行，拒绝重复并发启动") from None
    return handle


def status(run: Path, state: str, **extra) -> None:
    value = dict(status=state, updated_at=now(), pid=os.getpid(), **extra)
    save_json(run / "RUN_STATUS.json", value)
    labels = {
        "running": "运行中",
        "completed": "全部完成",
        "failed": "技术失败后停止",
        "paused": "已保存断点暂停"
    }
    lines = [
        "# 第七组洗发水与第九组牙刷“环境光背景协同调整＋主体微光影优化”（Ambient-Coop）实验状态",
        "",
        f"当前状态：{labels.get(state, state)}",
        f"更新时间：{now()}",
        f"进程 PID：{os.getpid()}",
        ""
    ]
    if "stage" in extra:
        lines.append(f"当前阶段：{extra['stage']}")
    if "group" in extra:
        lines.append(f"当前组别：{extra['group']}")
    if "profile" in extra:
        lines.append(f"当前档位：{extra['profile']}")
    if "log" in extra:
        lines.append(f"日志文件：{extra['log']}")
    lines += [
        "",
        "生成细节见各组目录 progress.json；实验完成后将自动生成 REPORT.md。",
        ""
    ]
    (run / "RUN_STATUS.md").write_text("\n".join(lines), encoding="utf-8")


def implementation_files() -> dict[str, str]:
    paths = [Path(__file__), PROJECT / "src/experiments/perlage_ambient_coop_evaluation.py"]
    amb_pkg = PROJECT / "src/visual/_perlage_ambient_coop"
    if amb_pkg.exists():
        paths += sorted(amb_pkg.glob("*.py"))
    paths += [PROJECT / p for p in (
        "src/visual/retrieval_objective.py", "src/visual/encoders/joint.py",
        "src/visual/encoders/clip.py", "src/visual/encoders/backbone.py",
        "src/visual/_perlage/components/latent_branch.py", "src/visual/_perlage/components/core.py",
        "src/visual/_perlage/components/colour.py", "src/visual/catalog_training_guard.py",
        "src/visual/visual_retrieval.py", "src/experiments/query_contract.py",
        "src/visual/target_features.py", "src/visual/robust_objectives.py"
    )]
    return {str(p): sha256_file(p) for p in paths if p.is_file()}


def output_root(path: Path | str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT / p
    run = p.resolve()
    if run.parent != PROJECT / "output" or run in (BASE, PREVIOUS_BASELINE, SHOPPING):
        raise ValueError("只允许在项目根输出目录的新独立实验目录写入")
    return run


def prepare(run: Path) -> dict:
    """准备阶段：仅读取公开训练材料与冻结前缀，严禁打开正式查询与记忆库。"""
    from src.visual._perlage_ambient_coop import AMBIENT_COOP_PRESETS
    run = output_root(run)
    run.mkdir(parents=True, exist_ok=True)
    groups = {}

    for key in GROUPS:
        gdir = BASE / "groups" / key
        config_path = gdir / "perlage_no_pixel_config.json"
        definition = read_json(config_path)
        definition["data"]["source"] = str(gdir / "source.jpg")
        definition["data"]["target"] = str(gdir / "target.jpg")

        public = read_json(gdir / "definition.json")
        source_id = public["source"]["id"]
        review_file = BASE / "public_metadata/product_feature_review.json"
        feature = read_json(review_file)["products"][source_id]
        catalog = read_json(BASE / "public_metadata/product_metadata.json")["products"][source_id]

        if feature["title"] != definition["data"]["source_title"] or public["source"]["title"] != feature["title"]:
            raise ValueError("公开源商品名不一致，禁止构造与商品无关的训练键")

        for evidence in feature["feature_evidence"]:
            field = catalog.get(evidence["field"], "")
            field = field[evidence["index"]] if "index" in evidence else field
            if evidence["quote"] not in str(field):
                raise ValueError("源商品特点缺少公开原文证据")

        source_record = dict(
            id=source_id,
            title=feature["title"],
            brand=feature["brand"],
            attributes=public["source"]["attributes"],
            feature_text=feature["canonical_sentence"],
            provenance=feature["source"],
            feature_evidence=feature["feature_evidence"]
        )

        if definition["data"]["training_queries"] != [definition["target_feature_text"]]:
            raise ValueError("训练必须逐字复用原目标特点正文")

        prefix = PREVIOUS_BASELINE / "groups" / key / "methods/PerLAGE_no_pixel/generation"
        bindings = {
            str(p): sha256_file(p) for p in (
                config_path, gdir / "definition.json", gdir / "source.jpg", gdir / "target.jpg",
                prefix / "ready.json", prefix / "adversarial.png",
                BASE / "public_metadata/product_metadata.json", review_file
            )
        }
        groups[key] = dict(
            definition=definition,
            source_public_record=source_record,
            prefix_root=str(prefix),
            files=bindings
        )

    pause = SHOPPING / "SHOPPING_PAUSED.json"
    checkpoint = SHOPPING / "USER_PAUSE_CHECKPOINT.json"
    preserved = {str(p): sha256_file(p) for p in (pause, checkpoint) if p.exists()}

    profiles = {
        p: asdict(AMBIENT_COOP_PRESETS[p]) if is_dataclass(AMBIENT_COOP_PRESETS[p]) else AMBIENT_COOP_PRESETS[p]
        for p in PROFILES
    }

    plan = dict(
        schema_version=1,
        experiment="perlage_ambient_coop_groups79_20261112",
        groups=list(GROUPS),
        profiles=list(PROFILES),
        profile_configs=profiles,
        generation_inputs=groups,
        implementation=implementation_files(),
        preserved_shopping_files=preserved,
        generation_seed={p: profiles[p].get("seed", 42) for p in PROFILES},
        training="original_frozen_single_target_feature_sentence",
        training_key="own_public_title_brand_attributes_feature_text_reconstructed_without_pool",
        query_mode="joint",
        condition="none",
        queries_per_group=20,
        query_image_weight=0.5,
        key_image_weight=0.5,
        shared_pool_count=189,
        all_images_frozen_before_evaluation=True,
        pixel_optimization=False,
        first_three_stages="verified_frozen_prefix_reused_not_newly_generated",
        spatial_envelope_policy="perlage_final_anchor_subject_micro_light_background_range_centered_ambient_smooth",
        smoothing_policy="fg_sigma_5.0_9.0_bg_sigma_2.2_4.5_ambient_sigma_14.0",
        generation_selection="training_objective_only_no_formal_ranking_selection",
        current_formal_rankings_used_for_parameter_selection=False,
        historical_group_results_seen_for_development=True,
        training_reads_shared_pool=False,
        shopping_enabled=False,
        model_api_calls_enabled=False,
        caveats=[
            "按用户要求测试第七组洗发水（07_shampoo）与第九组牙刷（09_toothbrush），共生成两张环境光背景协同完整图（ambient_coop）。",
            "商品主体锚定在 PerLAGE_no_pixel 最终成图，仅施加 6 级微光影柔化；背景在无截断可行区间中点同步联合优化全局环境底色、8x8 漫射网格与平滑 VAE 场（28 级），ROI 文字区 100% 零改动。",
            "训练代理与指定评估编码器重合，不声称独立留出黑盒迁移。",
            "沿用原训练正文；正式二十条特点文字有一条与训练精确重合。"
        ]
    )
    freeze_json(run / "EXPERIMENT_PLAN.json", plan)
    freeze_json(run / "PREPARED.json", dict(
        status="prepared", plan_sha256=sha256_file(run / "EXPERIMENT_PLAN.json"),
        formal_queries_read=False, shared_pool_read=False,
        generation_images=len(GROUPS) * len(PROFILES), shopping_enabled=False
    ))
    return plan


def validate_plan(run: Path) -> dict:
    plan = read_json(run / "EXPERIMENT_PLAN.json")
    if plan["groups"] != list(GROUPS) or plan["profiles"] != list(PROFILES) or plan["shopping_enabled"]:
        raise ValueError("本轮固定洗发水与牙刷双组、仅检索，实验范围改变")
    for path, expected in plan["implementation"].items():
        check(path, expected)
    for item in plan["generation_inputs"].values():
        for path, expected in item["files"].items():
            check(path, expected)
    for path, expected in plan["preserved_shopping_files"].items():
        check(path, expected)
    return plan


def image_root(run: Path, profile: str, key: str) -> Path:
    return run / "generation" / profile / key


def validate_image(run: Path, profile: str, key: str, plan: dict | None = None) -> dict:
    from src.visual._perlage_ambient_coop import validate_ambient_coop_complete
    root = image_root(run, profile, key)
    ready = read_json(root / "ready.json")
    validate_ambient_coop_complete(ready, root / "adversarial.png")

    plan = plan or read_json(run / "EXPERIMENT_PLAN.json")
    definition = plan["generation_inputs"][key]["definition"]
    config = plan["profile_configs"][profile]

    if (ready.get("profile") != profile or ready.get("latent_config") != config
            or ready["stages"]["latent_generation_refinement"]["steps_requested"] != config["steps"]):
        raise ValueError("完成图片档位或完整细化日程不符")

    for role in ("source", "target"):
        if ready.get(role + "_sha256") != sha256_file(definition["data"][role]):
            raise ValueError("完成图片绑定了错误组别的公开输入")

    identity = read_json(ready["method_identity"])
    if identity["public_definition"] != definition:
        raise ValueError("完成图片训练正文或分割保护配置改变")
    if identity["source_public_record"] != plan["generation_inputs"][key]["source_public_record"]:
        raise ValueError("完成图片自身公开特点全文键改变")

    return dict(
        image=str(root / "adversarial.png"),
        sha256=sha256_file(root / "adversarial.png"),
        ready=str(root / "ready.json"),
        ready_sha256=sha256_file(root / "ready.json")
    )


def generate(run: Path, profile: str, key: str) -> None:
    from src.visual.catalog_training_guard import catalog_training_guard
    from src.visual._perlage_ambient_coop import generate_ambient_coop

    plan = validate_plan(run)
    if (run / "PAUSE_REQUESTED").exists():
        raise InterruptedError("本轮已请求暂停")

    item = plan["generation_inputs"][key]
    definition = copy.deepcopy(item["definition"])
    root = image_root(run, profile, key)

    with catalog_training_guard(
        allowed_output_roots=[run, BASE / "groups" / key, Path(item["prefix_root"])],
        private_paths=[
            BASE / "shared_memory",
            BASE / "evaluation_inputs",
            SHOPPING,
            run / "retrieval",
            run / "banks",
            run / "EVALUATION_INPUTS.json"
        ]
    ) as audit:
        image, ready = generate_ambient_coop(
            definition["data"]["source"],
            definition["data"]["target"],
            root,
            definition=definition,
            prefix_root=item["prefix_root"],
            profile=profile,
            source_public_record=item["source_public_record"],
            device="cuda",
            pause_path=run / "PAUSE_REQUESTED"
        )

    freeze_json(root / "TRAINING_ACCESS_AUDIT.json", {k: v for k, v in audit.items() if k != "active"})
    if not image or ready.get("status") != "completed":
        raise InterruptedError("已保存生成断点")

    validate_image(run, profile, key, plan)
    print(json.dumps(dict(stage="generation_completed", profile=profile, group=key), ensure_ascii=False), flush=True)


def freeze_images(run: Path) -> None:
    validate_plan(run)
    images = {profile: {key: validate_image(run, profile, key) for key in GROUPS} for profile in PROFILES}
    freeze_json(run / "FROZEN_IMAGES.json", dict(
        status="frozen",
        images=images,
        plan_sha256=sha256_file(run / "EXPERIMENT_PLAN.json"),
        formal_rankings_used=False
    ))


def run_child(run: Path, stage: str, *, profile: str | None = None, group: str | None = None) -> None:
    if (run / "PAUSE_REQUESTED").exists():
        raise InterruptedError("本轮暂停标记存在")

    command = [
        sys.executable, "-X", "utf8", "-B", "-u", str(Path(__file__).resolve()),
        "--output", str(run), "--stage", stage
    ]
    if profile:
        command += ["--profile", profile, "--group", group]

    log = run / "logs" / ("_".join(x for x in (stage, profile, group) if x) + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    status(run, "running", stage=stage, profile=profile, group=group, log=str(log))

    with log.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command, cwd=PROJECT, stdout=handle, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
        )
        save_json(run / "ACTIVE_CHILD.json", dict(
            pid=process.pid, stage=stage, profile=profile,
            group=group, command=command, started_at=now()
        ))
        code = process.wait()

    if (run / "ACTIVE_CHILD.json").exists():
        try:
            (run / "ACTIVE_CHILD.json").unlink()
        except OSError:
            pass

    if code:
        if code == 2 and (run / "PAUSE_REQUESTED").exists():
            raise InterruptedError("工作进程已保存暂停断点")
        raise RuntimeError(f"阶段执行失败：{stage} {profile or ''} {group or ''}；退出码 {code}；日志见 {log}")


def queue(run: Path) -> None:
    """全自动无人值守执行流水线。"""
    plan = validate_plan(run)
    for profile in PROFILES:
        for key in GROUPS:
            root = image_root(run, profile, key)
            if (root / "ready.json").exists():
                validate_image(run, profile, key, plan)
            else:
                run_child(run, "generate", profile=profile, group=key)

    freeze_images(run)

    for stage, marker in (
        ("evaluate", "RETRIEVAL_COMPLETED.json"),
        ("verify", "INDEPENDENT_VERIFICATION.json"),
        ("report", "SUMMARY.json")
    ):
        run_child(run, stage)
        if not (run / marker).is_file():
            raise RuntimeError(f"缺少阶段完成证据文件：{marker}")

    validate_plan(run)
    inputs = read_json(run / "EVALUATION_INPUTS.json")

    freeze_json(run / "COMPLETED.json", dict(
        status="completed",
        generation_images=len(GROUPS) * len(PROFILES),
        retrieval_banks=len(inputs["bank_order"]),
        query_rankings=inputs["expected_rankings"],
        candidate_scores=inputs["expected_score_comparisons"],
        independent_verification=True,
        plan_sha256=sha256_file(run / "EXPERIMENT_PLAN.json"),
        frozen_images_sha256=sha256_file(run / "FROZEN_IMAGES.json"),
        verification_sha256=sha256_file(run / "INDEPENDENT_VERIFICATION.json"),
        summary_sha256=sha256_file(run / "SUMMARY.json"),
        report_sha256=sha256_file(run / "REPORT.md"),
        shopping_enabled=False
    ))
    status(run, "completed", report=str(run / "REPORT.md"), query_rankings=inputs["expected_rankings"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--all", action="store_true", help="等同于启动无人值守串行队列 (queue)")
    parser.add_argument("--stage", choices=("queue", "generate", "evaluate", "verify", "report"), default="queue")
    parser.add_argument("--profile", choices=PROFILES, default=PROFILES[0])
    parser.add_argument("--group", choices=GROUPS, default=GROUPS[0])
    args = parser.parse_args()

    run = output_root(args.output)
    run.mkdir(parents=True, exist_ok=True)

    stage = "queue" if args.all else args.stage
    lock_file = run / ("RUN.lock" if stage == "queue" else "WORKER.lock")
    handle = lock(lock_file)

    try:
        if args.prepare_only:
            prepare(run)
            print("洗发水与牙刷双组环境光背景协同实验计划已冻结；未打开正式图文查询和共享记忆。", flush=True)
            return

        if stage == "queue":
            if not (run / "EXPERIMENT_PLAN.json").exists():
                prepare(run)
            queue(run)
        elif stage == "generate":
            if not args.profile or not args.group:
                parser.error("生成阶段必须显式指定档位与组别 (--profile ambient_coop --group 07_shampoo)")
            generate(run, args.profile, args.group)
        else:
            validate_plan(run)
            freeze_images(run)
            from src.experiments import perlage_ambient_coop_evaluation as evaluation
            getattr(evaluation, stage)(run)
    except InterruptedError as error:
        status(run, "paused", stage=stage, message=str(error))
        raise SystemExit(2)
    except Exception as error:
        save_json(run / "FAILURE.json", dict(
            status="failed", stage=stage, profile=args.profile,
            group=args.group, error_type=type(error).__name__, message=str(error), failed_at=now()
        ))
        status(run, "failed", stage=stage, message=str(error))
        raise
    finally:
        handle.close()


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    main()
