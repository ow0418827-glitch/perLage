"""第六、七组无像素感知强攻实验：固定两档强攻、串行生成、冻结后图文检索。"""
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
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                  TOKENIZERS_PARALLELISM="false", PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1")

from src.visual.audit import freeze_json, read_json, save_json
from src.visual.image_utils import sha256_file

BASE = PROJECT / "output/experiment_base_groups9_20261109"
PREVIOUS = PROJECT / "output/perlage_no_pixel_groups9_20261109"
SHOPPING = PROJECT / "output/groups9_multimodal_20261110"
DEFAULT_RUN = PROJECT / "output/perlage_rankfocus_groups67_20261111"
GROUPS = ("06_jeans", "07_shampoo")
PROFILES = ("rankfocus", "rankfocus_wide")


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def check(path, expected):
    if sha256_file(path) != expected:
        raise ValueError("冻结文件摘要变化：" + str(path))


def lock(path):
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
        raise RuntimeError("已有本轮队列或工作进程，拒绝重复启动") from None
    return handle


def status(run, state, **extra):
    value = dict(status=state, updated_at=now(), pid=os.getpid(), **extra)
    save_json(run / "RUN_STATUS.json", value)
    labels = {"running": "运行中", "completed": "全部完成", "failed": "技术失败后停止", "paused": "已保存断点暂停"}
    lines = ["# 第六、七组无像素感知强攻实验", "", "当前状态：" + labels.get(state, state), ""]
    if "stage" in extra:
        lines.append("当前阶段：" + str(extra["stage"]))
    if "group" in extra:
        lines.append("当前组别：" + str(extra["group"]))
    if "profile" in extra:
        lines.append("当前档位：" + str(extra["profile"]))
    lines += ["", "逐步生成进度在各图片目录的 progress.json（进度记录）；完成后自动生成 REPORT.md（实验报告）。", ""]
    (run / "RUN_STATUS.md").write_text("\n".join(lines), encoding="utf-8")


def implementation_files():
    paths = [Path(__file__), PROJECT / "src/experiments/perlage_rankfocus_evaluation.py"]
    paths += sorted((PROJECT / "src/visual/_perlage_no_pixel_rankfocus").glob("*.py"))
    paths += sorted((PROJECT / "src/visual/_perlage_no_pixel_relaxed").glob("*.py"))
    paths += [PROJECT / p for p in (
        "src/visual/retrieval_objective.py", "src/visual/encoders/joint.py",
        "src/visual/encoders/clip.py", "src/visual/encoders/backbone.py",
        "src/visual/_perlage/components/latent_branch.py", "src/visual/_perlage/components/core.py",
        "src/visual/_perlage/components/colour.py", "src/visual/catalog_training_guard.py",
        "src/visual/visual_retrieval.py", "src/experiments/query_contract.py",
        "src/visual/target_features.py", "src/visual/robust_objectives.py")]
    return {str(p): sha256_file(p) for p in paths}


def output_root(path):
    run = Path(path).resolve()
    if run.parent != PROJECT / "output" or run in (BASE, PREVIOUS, SHOPPING):
        raise ValueError("只允许在项目根输出目录的新独立实验目录写入")
    return run


def prepare(run):
    """准备阶段仅读取公开训练材料及旧完整生成前缀，不打开正式查询和记忆库。"""
    from src.visual._perlage_no_pixel_rankfocus import RANKFOCUS_PRESETS
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
        source_record = dict(id=source_id, title=feature["title"], brand=feature["brand"],
            attributes=public["source"]["attributes"], feature_text=feature["canonical_sentence"],
            provenance=feature["source"], feature_evidence=feature["feature_evidence"])
        if definition["data"]["training_queries"] != [definition["target_feature_text"]]:
            raise ValueError("训练必须逐字复用原目标特点正文")
        prefix = PREVIOUS / "groups" / key / "methods/PerLAGE_no_pixel/generation"
        bindings = {str(p): sha256_file(p) for p in (config_path, gdir / "definition.json", gdir / "source.jpg", gdir / "target.jpg", prefix / "ready.json",
            BASE / "public_metadata/product_metadata.json", review_file)}
        groups[key] = dict(definition=definition, source_public_record=source_record,
            prefix_root=str(prefix), files=bindings)
    pause = SHOPPING / "SHOPPING_PAUSED.json"
    checkpoint = SHOPPING / "USER_PAUSE_CHECKPOINT.json"
    preserved = {str(p): sha256_file(p) for p in (pause, checkpoint) if p.exists()}
    profiles = {p: asdict(RANKFOCUS_PRESETS[p]) if is_dataclass(RANKFOCUS_PRESETS[p]) else RANKFOCUS_PRESETS[p] for p in PROFILES}
    plan = dict(schema_version=1, experiment="perlage_no_pixel_rankfocus_groups67_20261111",
        groups=list(GROUPS), profiles=list(PROFILES), profile_configs=profiles,
        generation_inputs=groups, implementation=implementation_files(), preserved_shopping_files=preserved,
        generation_seed={p: profiles[p]["seed"] for p in PROFILES}, training="original_frozen_single_target_feature_sentence",
        training_key="own_public_title_brand_attributes_feature_text_reconstructed_without_pool",
        previous_experiment_readonly="output/perlage_relaxed_groups67_20261111",
        query_mode="joint", condition="none", queries_per_group=20,
        query_image_weight=0.5, key_image_weight=0.5, shared_pool_count=189,
        all_images_frozen_before_evaluation=True, pixel_optimization=False,
        first_three_stages="verified_frozen_prefix_reused_not_newly_generated",
        generation_selection="training_objective_only_no_formal_ranking_selection",
        current_formal_rankings_used_for_parameter_selection=False,
        historical_group_results_seen_for_development=True, training_reads_shared_pool=False,
        shopping_enabled=False, model_api_calls_enabled=False,
        caveats=["按用户明确要求大幅放宽外观约束，但不保证显著改善；全部固定档位如实报告，不能按正式名次选最终图。",
            "训练代理与指定评估编码器重合，不声称独立留出黑盒迁移。",
            "沿用原训练正文；每组正式二十条特点文字有一条与训练精确重合。",
            "复用已完成前三阶段；本轮第四阶段重新完整执行1000步，初始化及颜色融合参数不变。",
            "放宽外观约束可能降低视觉保真；输出实际结构相似度、平均变化、印刷区和前景保护证据。"])
    freeze_json(run / "EXPERIMENT_PLAN.json", plan)
    freeze_json(run / "PREPARED.json", dict(status="prepared", plan_sha256=sha256_file(run / "EXPERIMENT_PLAN.json"),
        formal_queries_read=False, shared_pool_read=False, generation_images=4, shopping_enabled=False))
    return plan


def validate_plan(run):
    plan = read_json(run / "EXPERIMENT_PLAN.json")
    if plan["groups"] != list(GROUPS) or plan["profiles"] != list(PROFILES) or plan["shopping_enabled"]:
        raise ValueError("本轮固定两组两档、仅检索，实验范围改变")
    for path, expected in plan["implementation"].items():
        check(path, expected)
    for item in plan["generation_inputs"].values():
        for path, expected in item["files"].items():
            check(path, expected)
    for path, expected in plan["preserved_shopping_files"].items():
        check(path, expected)
    return plan


def image_root(run, profile, key):
    return run / "generation" / profile / key


def validate_image(run, profile, key, plan=None):
    from src.visual._perlage_no_pixel_rankfocus import validate_rankfocus_complete
    root = image_root(run, profile, key)
    ready = read_json(root / "ready.json")
    validate_rankfocus_complete(ready, root / "adversarial.png")
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
    return dict(image=str(root / "adversarial.png"), sha256=sha256_file(root / "adversarial.png"),
        ready=str(root / "ready.json"), ready_sha256=sha256_file(root / "ready.json"))


def generate(run, profile, key):
    from src.visual.catalog_training_guard import catalog_training_guard
    from src.visual._perlage_no_pixel_rankfocus import generate_rankfocus_no_pixel
    plan = validate_plan(run)
    if (run / "PAUSE_REQUESTED").exists():
        raise InterruptedError("本轮已请求暂停")
    item = plan["generation_inputs"][key]
    definition = copy.deepcopy(item["definition"])
    root = image_root(run, profile, key)
    with catalog_training_guard(allowed_output_roots=[run, BASE / "groups" / key, Path(item["prefix_root"])],
            private_paths=[BASE / "shared_memory", BASE / "evaluation_inputs", SHOPPING,
                run / "retrieval", run / "banks", run / "EVALUATION_INPUTS.json"]) as audit:
        image, ready = generate_rankfocus_no_pixel(definition["data"]["source"], definition["data"]["target"], root,
            definition=definition, prefix_root=item["prefix_root"], profile=profile,
            source_public_record=item["source_public_record"],
            device="cuda", pause_path=run / "PAUSE_REQUESTED")
    freeze_json(root / "TRAINING_ACCESS_AUDIT.json", {k: v for k, v in audit.items() if k != "active"})
    if not image or ready.get("status") != "completed":
        raise InterruptedError("已保存生成断点")
    validate_image(run, profile, key, plan)
    print(json.dumps(dict(stage="generation_completed", profile=profile, group=key), ensure_ascii=False), flush=True)


def freeze_images(run):
    validate_plan(run)
    images = {profile: {key: validate_image(run, profile, key) for key in GROUPS} for profile in PROFILES}
    freeze_json(run / "FROZEN_IMAGES.json", dict(status="frozen", images=images,
        plan_sha256=sha256_file(run / "EXPERIMENT_PLAN.json"), formal_rankings_used=False))


def run_child(run, stage, *, profile=None, group=None):
    if (run / "PAUSE_REQUESTED").exists():
        raise InterruptedError("本轮暂停标记存在")
    command = [sys.executable, "-X", "utf8", "-B", "-u", str(Path(__file__).resolve()),
        "--output", str(run), "--stage", stage]
    if profile:
        command += ["--profile", profile, "--group", group]
    log = run / "logs" / ("_".join(x for x in (stage, profile, group) if x) + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    status(run, "running", stage=stage, profile=profile, group=group, log=str(log))
    with log.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(command, cwd=PROJECT, stdout=handle, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        save_json(run / "ACTIVE_CHILD.json", dict(pid=process.pid, stage=stage, profile=profile,
            group=group, command=command, started_at=now()))
        code = process.wait()
    if code:
        if code == 2 and (run / "PAUSE_REQUESTED").exists():
            raise InterruptedError("工作进程已保存暂停断点")
        raise RuntimeError(f"阶段失败后停止：{stage} {profile or ''} {group or ''}；退出码{code}；日志{log}")


def queue(run):
    plan = validate_plan(run)
    for profile in PROFILES:
        for key in GROUPS:
            root = image_root(run, profile, key)
            if (root / "ready.json").exists():
                validate_image(run, profile, key, plan)
            else:
                run_child(run, "generate", profile=profile, group=key)
    freeze_images(run)
    for stage, marker in (("evaluate", "RETRIEVAL_COMPLETED.json"), ("verify", "INDEPENDENT_VERIFICATION.json"), ("report", "SUMMARY.json")):
        # 阶段函数本身核验并复用冻结证据；已完成阶段也在独立进程重新校验。
        run_child(run, stage)
        if not (run / marker).is_file():
            raise RuntimeError("缺少阶段完成证据：" + marker)
    validate_plan(run)
    freeze_json(run / "COMPLETED.json", dict(status="completed", generation_images=4, retrieval_banks=6,
        query_rankings=240, candidate_scores=45360, independent_verification=True,
        plan_sha256=sha256_file(run / "EXPERIMENT_PLAN.json"),
        frozen_images_sha256=sha256_file(run / "FROZEN_IMAGES.json"),
        verification_sha256=sha256_file(run / "INDEPENDENT_VERIFICATION.json"),
        summary_sha256=sha256_file(run / "SUMMARY.json"), report_sha256=sha256_file(run / "REPORT.md"),
        shopping_enabled=False))
    status(run, "completed", report=str(run / "REPORT.md"), query_rankings=240)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--stage", choices=("queue", "generate", "evaluate", "verify", "report"), default="queue")
    parser.add_argument("--profile", choices=PROFILES)
    parser.add_argument("--group", choices=GROUPS)
    args = parser.parse_args()
    run = output_root(args.output)
    run.mkdir(parents=True, exist_ok=True)
    handle = lock(run / ("RUN.lock" if args.stage == "queue" else "WORKER.lock"))
    try:
        if args.prepare_only:
            prepare(run)
            print("两组两档训练计划已冻结；未打开正式询问和共享记忆。", flush=True)
            return
        if args.stage == "queue":
            if not (run / "EXPERIMENT_PLAN.json").exists():
                prepare(run)
            queue(run)
        elif args.stage == "generate":
            if not args.profile or not args.group:
                parser.error("生成阶段必须指定档位与组别")
            generate(run, args.profile, args.group)
        else:
            validate_plan(run)
            freeze_images(run)
            from src.experiments import perlage_rankfocus_evaluation as evaluation
            getattr(evaluation, args.stage)(run)
    except InterruptedError as error:
        status(run, "paused", stage=args.stage, message=str(error))
        raise SystemExit(2)
    except Exception as error:
        save_json(run / "FAILURE.json", dict(status="failed", stage=args.stage, profile=args.profile,
            group=args.group, error_type=type(error).__name__, message=str(error), failed_at=now()))
        status(run, "failed", stage=args.stage, message=str(error))
        raise
    finally:
        handle.close()


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    main()
