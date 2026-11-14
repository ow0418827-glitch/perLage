"""第七组洗发水与第九组牙刷“环境光背景协同调整＋主体微光影优化”（Ambient-Coop）实验：
两组完整图片全部冻结后，在 189 条共享长期记忆池中进行同步双图替换图文检索评测、
独立子进程全量分数复算与综合对比报告生成。
"""
from __future__ import annotations

import copy
import math
import os

import numpy as np
import torch

from src.paths import PROJECT, EXPERIMENT_BASE_GROUPS9 as BASE, ProjectPath as Path
from src.visual.audit import digest, freeze_json, read_json
from src.visual.image_utils import sha256_file
from src.visual._perlage.components.colour import read_rgb
from src.visual._perlage.components.core import source_ssim


GROUPS = ("07_shampoo", "09_toothbrush")
PROFILES = ("ambient_coop",)
BANKS = ("clean", "baseline", "lucid", *PROFILES)

# 历史只读基线路径
PREVIOUS_BASELINE = PROJECT / "output/perlage_no_pixel_groups9_20261109"
LUCID_RUN = PROJECT / "output/lucid_groups9_20261109"
PREVIOUS_DUAL_SMOOTH = PROJECT / "output/perlage_dual_smooth_shampoo_20261112"
QUERY_RUN = PROJECT / "output/groups9_multimodal_20261110"

STAGES = ("foreground_segmentation", "foreground_statistics", "color_harmonization", "latent_generation_refinement")


def _check(path: str | Path, expected: str) -> None:
    path_obj = Path(path)
    if not expected or not path_obj.is_file() or sha256_file(path_obj) != expected:
        raise ValueError(f"冻结文件缺失或摘要改变：{path}")


def _frozen_generation(run: Path | str) -> dict:
    """在两组图片生成四阶段凭证与冻结检验完成前，严禁读取共享池与正式查询。"""
    run = Path(run).resolve()
    plan_file = run / "EXPERIMENT_PLAN.json"
    plan = read_json(plan_file)
    if plan.get("groups") != list(GROUPS) or plan.get("profiles") != list(PROFILES):
        raise ValueError("实验必须预先冻结第七组洗发水、第九组牙刷及 ambient_coop 环境光协同档位")

    registry_file = run / "FROZEN_IMAGES.json"
    frozen = read_json(registry_file)
    if frozen.get("status") != "frozen" or set(frozen.get("images", {})) != set(PROFILES):
        raise ValueError("环境光协同档位完整图片必须全部冻结后方可评测")
    if frozen.get("plan_sha256") != sha256_file(plan_file):
        raise ValueError("冻结图片未绑定预定训练计划")

    for profile in PROFILES:
        if set(frozen["images"][profile]) != set(GROUPS):
            raise ValueError("冻结图的组别范围不完整")
        for key in GROUPS:
            item = frozen["images"][profile][key]
            _check(item["image"], item["sha256"])
            _check(item["ready"], item["ready_sha256"])
            ready = read_json(item["ready"])

            from src.visual._perlage_ambient_coop import validate_ambient_coop_complete
            validate_ambient_coop_complete(ready, item["image"])

            expected_config = plan["profile_configs"][profile]
            if ready.get("profile") != profile or ready.get("latent_config") != expected_config:
                raise ValueError("完整成图没有绑定预注册档位配置")
            if expected_config.get("steps") != 350:
                raise ValueError("细化步数必须为 350 步")
            if not (0 <= expected_config.get("latent_levels", 0) <= 12):
                raise ValueError("主体微光影预算不符合方案约束 (0~12级)")
            if not (16 <= expected_config.get("background_levels", 0) <= 36):
                raise ValueError("背景环境光预算不符合方案约束 (16~36级)")

            public_data = plan["generation_inputs"][key]["definition"]["data"]
            for role in ("source", "target"):
                if ready.get(role + "_sha256") != sha256_file(public_data[role]):
                    raise ValueError(f"完整成图与预注册公开图对不一致：{role}")

            if ready.get("status") != "completed" or ready.get("pixel_optimization") is not False:
                raise ValueError("实验要求完整无像素优化成图")
            if any(ready.get(f) is not False for f in
                   ("formal_queries_used", "victim_memory_used", "shadow_pool_used", "purchase_history_used")):
                raise ValueError("生成凭证未明确排除正式查询、记忆和影子库训练")
            if ready.get("auxiliary_max_levels", 0) != 0 or ready.get("latent_config", {}).get("auxiliary_levels", 0) != 0:
                raise ValueError("环境光协同实验严禁添加像素残差")

            stages = ready.get("stages", {})
            if set(stages) != set(STAGES):
                raise ValueError("成图必须保留完整四阶段")
            for stage in STAGES:
                record = stages[stage]
                if record.get("status") != "completed" or not record.get("files"):
                    raise ValueError(f"生成阶段证据不完整：{stage}")
                for p, exp in record["files"].items():
                    _check(p, exp)

            latent = stages["latent_generation_refinement"]
            if (type(latent.get("steps_completed")) is not int or latent["steps_completed"] < 1
                    or latent["steps_completed"] != latent.get("steps_requested")):
                raise ValueError("最终潜变量细化未完成预定步数")

            final = ready.get("candidates", {}).get("final", {})
            if final.get("sha256") != item["sha256"]:
                raise ValueError("冻结图与完整方法最终候选图片摘要不一致")
    return frozen


def build_banks(entries: list[dict], pool_path_or_replacements: Path | str | dict, replacements: dict[str, dict[str, dict]] | None = None) -> dict[str, list[dict]]:
    """保留完整 189 条共享池；每个修改库同步替换洗发水与牙刷对应源记忆图片。"""
    if replacements is None:
        pool_path = None
        replacements = pool_path_or_replacements
    else:
        pool_path = pool_path_or_replacements

    if len(entries) != 189 or len({e["Id"] for e in entries}) != 189:
        raise ValueError("必须使用完整且唯一的一百八十九条共享池")

    target_ids = {
        "groups9_reference_07_shampoo_B08XYBS678",
        "groups9_reference_09_toothbrush_B09PNNXYJ4",
    }
    for bank_name, change_map in replacements.items():
        if set(change_map.keys()) != target_ids:
            raise ValueError(f"每个修改库必须同步替换洗发水与牙刷两个条目: 图库 {bank_name} 替换集不匹配")

    clean = copy.deepcopy(entries)
    for entry in clean:
        if entry.get("RetrievalImage") or entry.get("ImageSubstitution"):
            raise ValueError("基础共享池不能包含历史图片替换标记")
        if pool_path is not None:
            from src.visual.visual_retrieval import resolve_image
            image = resolve_image(entry, pool_path)
            _check(image, entry["PrimaryImage"]["SHA256"])
            entry["PrimaryImage"]["LocalPath"] = str(Path(image).resolve())
        if entry.get("RecordKind") == "simulated_gallery" and (entry.get("Completed") is True or entry.get("Actions")):
            raise ValueError("未购买参考不能含伪造购买轨迹")

    banks = {"clean": clean}
    for name, changes in replacements.items():
        bank = copy.deepcopy(clean)
        for entry in bank:
            if entry["Id"] in changes:
                asset = changes[entry["Id"]]
                if pool_path is not None:
                    _check(asset["image"], asset["sha256"])
                entry["RetrievalImage"] = dict(
                    ASIN=entry.get("FinalASIN", ""),
                    LocalPath=str(Path(asset["image"]).resolve()),
                    SHA256=asset["sha256"]
                )
        stripped = [{k: v for k, v in entry.items() if k != "RetrievalImage"} for entry in bank]
        if stripped != clean or sum("RetrievalImage" in entry for entry in bank) != len(GROUPS):
            raise ValueError(f"实验库 {name} 改变了图片以外的内容或替换范围不为 {len(GROUPS)}")
        banks[name] = bank
    return banks


def _prepare_inputs(run: Path) -> dict:
    frozen = _frozen_generation(run)
    from src.agent.memory.shared_pool import load_shared_pool
    from src.experiments.query_contract import load_formal_queries
    from src.visual.visual_retrieval import detect_index_text_policy

    root = read_json(BASE / "manifest.json")
    pool_path = BASE / root["shared_memory"]["pool_file"]
    _check(pool_path, root["shared_memory"]["pool_sha256"])
    entries = load_shared_pool(pool_path)

    encoder_path = BASE / root["evaluation_protocol"]["encoder_config"]
    _check(encoder_path, root["evaluation_protocol"]["encoder_config_sha256"])

    bindings = {
        str(BASE / "manifest.json"): sha256_file(BASE / "manifest.json"),
        str(pool_path): sha256_file(pool_path),
        str(encoder_path): sha256_file(encoder_path),
        str(run / "EXPERIMENT_PLAN.json"): sha256_file(run / "EXPERIMENT_PLAN.json"),
        str(run / "FROZEN_IMAGES.json"): sha256_file(run / "FROZEN_IMAGES.json"),
    }

    groups = {}
    replacements = {name: {} for name in BANKS if name != "clean"}
    expected_mem_ids = {
        "07_shampoo": "groups9_reference_07_shampoo_B08XYBS678",
        "09_toothbrush": "groups9_reference_09_toothbrush_B09PNNXYJ4",
    }

    for key in GROUPS:
        meta = root["groups"][key]
        source_id = root["shared_memory"]["source_mapping"][key]["source_memory_id"]
        if source_id != expected_mem_ids[key]:
            raise ValueError(f"源记忆映射不匹配：{source_id} != {expected_mem_ids[key]}")

        # 1. baseline (PerLAGE_no_pixel)
        orig_file = PREVIOUS_BASELINE / "groups" / key / "methods/PerLAGE_no_pixel/FROZEN_IMAGE.json"
        orig = read_json(orig_file)
        _check(orig["image"], orig["image_sha256"])
        bindings[str(orig_file)] = sha256_file(orig_file)
        bindings[orig["image"]] = orig["image_sha256"]
        replacements["baseline"][source_id] = dict(image=orig["image"], sha256=orig["image_sha256"])

        # 2. lucid
        lucid_file = LUCID_RUN / "groups" / key / "methods/lucid/FROZEN_IMAGE.json"
        lucid = read_json(lucid_file)
        _check(lucid["image"], lucid["image_sha256"])
        bindings[str(lucid_file)] = sha256_file(lucid_file)
        bindings[lucid["image"]] = lucid["image_sha256"]
        replacements["lucid"][source_id] = dict(image=lucid["image"], sha256=lucid["image_sha256"])

        # 3. ambient_coop (本轮新生成两组环境光协同图)
        for profile in PROFILES:
            asset = frozen["images"][profile][key]
            ready = read_json(asset["ready"])
            if ready.get("source_sha256") != meta["source_sha256"] or ready.get("target_sha256") != meta["target_sha256"]:
                raise ValueError("环境光协同完整图与本轮冻结公开图对不一致")
            replacements[profile][source_id] = dict(image=asset["image"], sha256=asset["sha256"])
            bindings[asset["image"]] = asset["sha256"]
            bindings[asset["ready"]] = asset["ready_sha256"]

        # 正式图文查询绑定（严格图文联合，query_mode="joint"）
        manifest_file = QUERY_RUN / "input_resources/formal_queries" / key / "queries.json"
        query_document = read_json(manifest_file)
        text_file = BASE / meta["query_manifest"]
        _check(text_file, meta["query_manifest_sha256"])
        text_doc = read_json(text_file)
        if query_document.get("instructions") != text_doc["instructions"]:
            raise ValueError("图文检索必须逐字复用既有特点文字")
        queries = load_formal_queries(manifest_file, orig["target"], query_mode="joint")
        if len(queries) != 20:
            raise ValueError("每组必须保留二十条正式图文请求")
        bindings[str(manifest_file)] = sha256_file(manifest_file)
        bindings[str(text_file)] = sha256_file(text_file)
        for query in queries:
            bindings[query["query_image"]] = query["query_image_sha256"]

        groups[key] = dict(
            name=meta["name"],
            source_id=meta["source_id"],
            target_id=meta["target_id"],
            source_memory_id=source_id,
            queries=queries,
            training_overlap=query_document.get("training_overlap")
        )

    banks = build_banks(entries, pool_path, replacements)
    paths = {}
    for name in BANKS:
        path = run / "banks" / f"{name}.json"
        freeze_json(path, banks[name])
        paths[name] = dict(path=str(path), sha256=sha256_file(path))
        bindings[str(path)] = sha256_file(path)

    expected_rankings = len(BANKS) * len(GROUPS) * 20
    expected_score_comparisons = expected_rankings * 189

    inputs = dict(
        schema_version=1,
        groups=groups,
        bank_order=list(BANKS),
        pool_count=189,
        banks=paths,
        encoder_config=str(encoder_path),
        encoder_config_sha256=sha256_file(encoder_path),
        bindings=bindings,
        query_mode="joint",
        query_image_weight=0.5,
        key_image_weight=0.5,
        index_text=detect_index_text_policy(entries),
        condition="none",
        shopping_allowed=False,
        simultaneous_replacements_per_modified_bank=len(GROUPS),
        expected_rankings=expected_rankings,
        expected_score_comparisons=expected_score_comparisons,
        baseline_policy="recompute_all_reference_banks_in_same_complete_pool",
        current_round_rankings_used_for_selection=False,
        historical_group_results_seen_for_development=True,
        input_read_only_after_all_complete_images=True
    )
    freeze_json(run / "EVALUATION_INPUTS.json", inputs)
    return inputs


def _inputs(run: Path) -> dict:
    _frozen_generation(run)
    inputs = read_json(run / "EVALUATION_INPUTS.json")
    if (inputs.get("bank_order") != list(BANKS) or set(inputs.get("groups", {})) != set(GROUPS)
            or inputs.get("pool_count") != 189 or inputs.get("shopping_allowed") is not False):
        raise ValueError("冻结评估范围不符")
    for path, expected in inputs["bindings"].items():
        _check(path, expected)
    return inputs


class _MemoizedEncoder:
    """仅在当前独立进程内存中复用本次编码向量；绝不加载任何磁盘向量缓存。"""
    supports_image_weight = True

    def __init__(self, encoder):
        self.encoder, self.snapshot = encoder, encoder.snapshot
        self.cache, self.last_encoding_audit = {}, None

    def encode_joint(self, text: str, image: str | Path, image_weight: float = 0.5):
        key = (text, sha256_file(image), image_weight)
        if key not in self.cache:
            vector = self.encoder.encode_joint(text, image, image_weight=image_weight)
            self.cache[key] = (vector, copy.deepcopy(getattr(self.encoder, "last_encoding_audit", None)))
        vector, audit = self.cache[key]
        self.last_encoding_audit = copy.deepcopy(audit)
        return vector


def _ranker(encoder, run: Path, inputs: dict, name: str):
    from src.visual.visual_retrieval import VisualMemoryRetriever
    bank = read_json(inputs["banks"][name]["path"])
    if len(bank) != 189:
        raise ValueError("检索库未覆盖全部一百八十九条")
    ranker = VisualMemoryRetriever(
        encoder, inputs["banks"][name]["path"], cache_dir=None,
        query_mode="joint", key_image_weight=0.5, query_image_weight=0.5,
        index_text=inputs["index_text"]
    )
    return bank, ranker


def _metrics(ranks: list[int]) -> dict:
    count = len(ranks)
    if not count or any(type(r) is not int or not 1 <= r <= 189 for r in ranks):
        raise ValueError("正式排名范围无效")
    return dict(
        query_count=count,
        mean_rank=sum(ranks) / count,
        mrr=sum(1.0 / r for r in ranks) / count,
        **{f"top{k}": sum(r <= k for r in ranks) for k in (1, 3, 5)},
        **{f"top{k}_rate": sum(r <= k for r in ranks) / count for k in (1, 3, 5)}
    )


def _result_path(run: Path, name: str, key: str) -> Path:
    return run / "retrieval" / name / f"{key}.json"


def _validate_result(run: Path, inputs: dict, name: str, key: str) -> dict:
    result = read_json(_result_path(run, name, key))
    protocol = result.get("protocol", {})
    if (result.get("inputs_sha256") != sha256_file(run / "EVALUATION_INPUTS.json")
            or result.get("bank") != name or result.get("group") != key
            or protocol.get("query_mode") != "joint" or protocol.get("key_mode") != "joint"
            or protocol.get("key_image_weight") != 0.5 or protocol.get("query_image_weight") != 0.5
            or protocol.get("bank_sha256") != inputs["banks"][name]["sha256"]):
        raise ValueError("既有排名未绑定当前完整图文评估条件")

    queries, rows = inputs["groups"][key]["queries"], result.get("rankings", [])
    bank_ids = {e["Id"] for e in read_json(inputs["banks"][name]["path"])}
    if len(rows) != 20 or [r.get("task_id") for r in rows] != [q["task_id"] for q in queries]:
        raise ValueError("既有排名未完整覆盖同样二十条正式请求")

    for row in rows:
        full = row.get("ranking", [])
        if (len(full) != 189 or {r["Id"] for r in full} != bank_ids
                or [r["rank"] for r in full] != list(range(1, 190))
                or any(not math.isfinite(r["score"]) for r in full)
                or any(a["score"] < b["score"] for a, b in zip(full, full[1:]))):
            raise ValueError("既有排名未完整保存全库有序候选分数")
        source = next(r for r in full if r["Id"] == inputs["groups"][key]["source_memory_id"])
        if row.get("source_rank") != source["rank"] or row.get("source_score") != source["score"]:
            raise ValueError("既有源记忆结果与完整排名不一致")

    if result.get("metrics") != _metrics([row["source_rank"] for row in rows]):
        raise ValueError("既有指标与正式名次不一致")
    return result


def _validate_retrieval(run: Path, inputs: dict) -> dict:
    certificate = read_json(run / "RETRIEVAL_COMPLETED.json")
    expected_files = {str(_result_path(run, n, k)) for n in BANKS for k in GROUPS}
    exp_rankings = inputs["expected_rankings"]
    exp_scores = inputs["expected_score_comparisons"]

    if (certificate.get("status") != "completed" or certificate.get("rankings") != exp_rankings
            or certificate.get("score_count") != exp_scores or certificate.get("execution_device") != "cuda"
            or certificate.get("inputs_sha256") != sha256_file(run / "EVALUATION_INPUTS.json")
            or set(certificate.get("files", {})) != expected_files):
        raise ValueError("检索完成证书范围或输入绑定不完整")

    for path, expected in certificate["files"].items():
        _check(path, expected)
    for name in BANKS:
        for key in GROUPS:
            _validate_result(run, inputs, name, key)
    return certificate


def _validate_verification(run: Path, inputs: dict) -> dict:
    retrieval = _validate_retrieval(run, inputs)
    verification = read_json(run / "INDEPENDENT_VERIFICATION.json")
    exp_rankings = inputs["expected_rankings"]
    exp_scores = inputs["expected_score_comparisons"]

    if (verification.get("status") != "passed" or verification.get("score_comparisons") != exp_scores
            or verification.get("rankings_recomputed") != exp_rankings or verification.get("fresh_process") is not True
            or verification.get("fresh_encoder") is not True or verification.get("execution_device") != "cuda"
            or verification.get("pid") == retrieval.get("pid")
            or verification.get("retrieval_pid") != retrieval.get("pid")
            or verification.get("files") != retrieval.get("files")
            or verification.get("inputs_sha256") != sha256_file(run / "EVALUATION_INPUTS.json")
            or not isinstance(verification.get("max_score_error"), (int, float))
            or not math.isfinite(verification["max_score_error"]) or verification["max_score_error"] > 1e-6):
        raise ValueError("独立复算证书未绑定本轮全部候选分数或误差超标")

    _check(run / "RETRIEVAL_COMPLETED.json", verification["retrieval_certificate_sha256"])
    return verification


def evaluate(run: Path | str) -> None:
    """两组完整图片全部冻结后，新编码所有全局库与两组各 20 条图文查询。"""
    run = Path(run).resolve()
    inputs = _inputs(run) if (run / "EVALUATION_INPUTS.json").is_file() else _prepare_inputs(run)
    if (run / "RETRIEVAL_COMPLETED.json").is_file():
        _validate_retrieval(run, inputs)
        return

    from src.experiments.shared_evaluation import make_eval_encoder
    from src.experiments.query_contract import verify_query_input

    raw = make_eval_encoder("cuda", inputs["encoder_config"])
    encoder = _MemoizedEncoder(raw)
    try:
        for name in BANKS:
            bank, ranker = _ranker(encoder, run, inputs, name)
            for key in GROUPS:
                path = _result_path(run, name, key)
                if path.exists():
                    _validate_result(run, inputs, name, key)
                    continue
                rows = []
                for query in inputs["groups"][key]["queries"]:
                    verify_query_input(query)
                    ranking = ranker.rank(bank, query["instruction"], query["query_image"])
                    if len(ranking) != 189 or any(not math.isfinite(r["score"]) for r in ranking):
                        raise ValueError("完整检索分数无效")
                    source = next(r for r in ranking if r["Id"] == inputs["groups"][key]["source_memory_id"])
                    rows.append(dict(
                        task_id=query["task_id"],
                        source_rank=source["rank"],
                        source_score=source["score"],
                        ranking=ranking
                    ))
                freeze_json(path, dict(
                    bank=name, group=key, protocol=copy.deepcopy(ranker.snapshot),
                    rankings=rows, metrics=_metrics([row["source_rank"] for row in rows]),
                    inputs_sha256=sha256_file(run / "EVALUATION_INPUTS.json")
                ))
                print(f"检索完成：{name} {key}，二十请求、完整一百八十九条库", flush=True)
    finally:
        raw.close()

    files = {str(_result_path(run, n, k)): sha256_file(_result_path(run, n, k)) for n in BANKS for k in GROUPS}
    freeze_json(run / "RETRIEVAL_COMPLETED.json", dict(
        status="completed", pid=os.getpid(), execution_device="cuda",
        inputs_sha256=sha256_file(run / "EVALUATION_INPUTS.json"),
        rankings=inputs["expected_rankings"], score_count=inputs["expected_score_comparisons"],
        persistent_vector_cache_read=False, files=files
    ))


def verify(run: Path | str) -> None:
    """必须在新进程执行；从零编码全部候选，逐项核对每一候选身份、名次与浮点分数。"""
    run = Path(run).resolve()
    inputs = _inputs(run)
    retrieval = _validate_retrieval(run, inputs)
    if (run / "INDEPENDENT_VERIFICATION.json").is_file():
        _validate_verification(run, inputs)
        return

    if retrieval.get("pid") == os.getpid():
        raise ValueError("独立复算必须使用不同进程")
    if retrieval.get("inputs_sha256") != sha256_file(run / "EVALUATION_INPUTS.json"):
        raise ValueError("检索完成证书未绑定本轮评估输入")
    for path, expected in retrieval["files"].items():
        _check(path, expected)

    from src.experiments.shared_evaluation import make_eval_encoder
    from src.experiments.query_contract import verify_query_input

    raw = make_eval_encoder("cuda", inputs["encoder_config"])
    encoder = _MemoizedEncoder(raw)
    compared, rankings, maximum = 0, 0, 0.0
    try:
        for name in BANKS:
            bank, ranker = _ranker(encoder, run, inputs, name)
            for key in GROUPS:
                old = read_json(_result_path(run, name, key))
                if digest(encoder.snapshot) != old["protocol"]["model_sha256"] or len(old["rankings"]) != 20:
                    raise ValueError("独立复算编码器或请求数量改变")
                for query, previous in zip(inputs["groups"][key]["queries"], old["rankings"]):
                    verify_query_input(query)
                    actual = ranker.rank(bank, query["instruction"], query["query_image"])
                    if query["task_id"] != previous["task_id"] or len(actual) != len(previous["ranking"]) or len(actual) != 189:
                        raise ValueError("独立复算请求身份或候选数不一致")
                    for a, b in zip(actual, previous["ranking"]):
                        if any(a[field] != b[field] for field in
                               ("Id", "rank", "image_sha256", "text_sha256", "query_text_sha256", "query_image_sha256")):
                            raise ValueError("独立复算候选身份、图片、文字或名次不一致")
                        error = abs(a["score"] - b["score"])
                        if not math.isfinite(error) or error > 1e-6:
                            raise ValueError(f"独立复算候选分数差异超过容差：{error:.9g} > 1e-6")
                        compared, maximum = compared + 1, max(maximum, error)
                    source = next(r for r in actual if r["Id"] == inputs["groups"][key]["source_memory_id"])
                    if source["rank"] != previous["source_rank"] or abs(source["score"] - previous["source_score"]) > 1e-6:
                        raise ValueError("保存的源记忆名次或分数与完整排名不符")
                    rankings += 1
                if old["metrics"] != _metrics([r["source_rank"] for r in old["rankings"]]):
                    raise ValueError("原始排名与汇总指标不一致")
    finally:
        raw.close()

    if compared != inputs["expected_score_comparisons"] or rankings != inputs["expected_rankings"]:
        raise ValueError(f"独立复算未覆盖全部预期比较：rankings={rankings}, compared={compared}")

    freeze_json(run / "INDEPENDENT_VERIFICATION.json", dict(
        status="passed", fresh_process=True, pid=os.getpid(),
        retrieval_pid=retrieval["pid"], fresh_encoder=True, execution_device="cuda",
        rankings_recomputed=rankings, score_comparisons=compared, max_score_error=maximum,
        persistent_vector_cache_read=False, inputs_sha256=sha256_file(run / "EVALUATION_INPUTS.json"),
        retrieval_certificate_sha256=sha256_file(run / "RETRIEVAL_COMPLETED.json"),
        files=retrieval["files"]
    ))


def _paired(current: list[dict], previous: list[dict]) -> dict:
    if [r["task_id"] for r in current] != [r["task_id"] for r in previous]:
        raise ValueError("配对名次的正式请求不一致")
    pairs = [(a["source_rank"], b["source_rank"]) for a, b in zip(current, previous)]
    return dict(
        top3_gained=sum(a <= 3 < b for a, b in pairs),
        top3_lost=sum(b <= 3 < a for a, b in pairs),
        top5_gained=sum(a <= 5 < b for a, b in pairs),
        top5_lost=sum(b <= 5 < a for a, b in pairs),
        improved_ranks=sum(a < b for a, b in pairs),
        worsened_ranks=sum(a > b for a, b in pairs),
        tied_ranks=sum(a == b for a, b in pairs),
    )


def _compute_baseline_visual(key: str) -> dict:
    """从冻结 PerLAGE_no_pixel 产物与源原图直接计算分区与全图视觉指标。"""
    gen_dir = PREVIOUS_BASELINE / "groups" / key / "methods/PerLAGE_no_pixel/generation"
    meta = read_json(gen_dir / "latent_generation_refinement/metadata.json")
    source_rgb = read_rgb(BASE / "groups" / key / "source.jpg")
    final_rgb = read_rgb(gen_dir / "adversarial.png")
    fg_mask = read_rgb(gen_dir / "foreground_segmentation/source/foreground.png")[..., 0] > 0
    diff = np.abs(final_rgb.astype(np.float32) - source_rgb.astype(np.float32))
    t_final = torch.as_tensor(final_rgb.transpose(2, 0, 1), dtype=torch.float32)[None] / 255.0
    t_source = torch.as_tensor(source_rgb.transpose(2, 0, 1), dtype=torch.float32)[None] / 255.0
    fc = meta.get("final_check", {})
    return dict(
        steps=meta.get("steps_completed", meta.get("config", {}).get("steps", 60)),
        best_step=fc.get("step", meta.get("steps_completed", 60)),
        accepted=meta.get("accepted_updates", 0),
        rejected=meta.get("rejected_updates", 0),
        source_ssim=float(source_ssim(t_final, t_source)),
        source_mae_levels=float(diff.mean()),
        fg_source_mae_levels=float(diff[fg_mask].mean()),
        bg_source_mae_levels=float(diff[~fg_mask].mean()),
        fg_latent_max_levels=0,
        fg_latent_mae_levels=0.0,
        bg_latent_max_levels=0,
        bg_latent_mae_levels=0.0,
        bg_mean_rgb_shift_levels=[0.0, 0.0, 0.0],
        halo_error=float(fc.get("halo_error", 0.0)),
        joint_score=float(fc.get("selection", [0.0])[-1]) if isinstance(fc.get("selection"), list) else 0.0,
    )


def _visual_summary(run: Path) -> dict:
    """提取本轮 ambient_coop 与基线的外观指标供综合报告对比。"""
    visual = {}
    for profile in PROFILES:
        visual[profile] = {}
        for key in GROUPS:
            meta = read_json(run / "generation" / profile / key / "metadata.json")
            fc = meta["final_check"]
            visual[profile][key] = dict(
                steps=meta["steps_completed"],
                best_step=meta["best"]["step"],
                accepted=meta["accepted_updates"],
                rejected=meta["rejected_updates"],
                source_ssim=fc["source_ssim"],
                source_mae_levels=fc["source_mae_levels"],
                fg_source_mae_levels=fc.get("fg_source_mae_levels"),
                bg_source_mae_levels=fc.get("bg_source_mae_levels"),
                fg_anchor_mae_levels=fc.get("fg_anchor_mae_levels"),
                bg_anchor_mae_levels=fc.get("bg_anchor_mae_levels"),
                fg_latent_max_levels=meta.get("fg_latent_max_levels"),
                fg_latent_mae_levels=meta.get("fg_latent_mae_levels"),
                bg_latent_max_levels=meta.get("bg_latent_max_levels"),
                bg_latent_mae_levels=meta.get("bg_latent_mae_levels"),
                bg_mean_rgb_shift_levels=fc.get("bg_mean_rgb_shift_levels", [0.0, 0.0, 0.0]),
                halo_error=fc["halo_error"],
                joint_score=fc["joint_scores"][0][0],
            )

    visual["baseline"] = {key: _compute_baseline_visual(key) for key in GROUPS}
    return visual


def report(run: Path | str) -> dict:
    """汇总评测指标，输出 SUMMARY.json 与 REPORT.md。"""
    run = Path(run).resolve()
    inputs = _inputs(run)
    verification = _validate_verification(run, inputs)
    results = {name: {key: read_json(_result_path(run, name, key)) for key in GROUPS} for name in BANKS}
    visual = _visual_summary(run)

    # 读取洗发水历史消融汇总供单组纵向参考（如存在）
    shampoo_history = {}
    if (PREVIOUS_DUAL_SMOOTH / "SUMMARY.json").is_file():
        prev_sum = read_json(PREVIOUS_DUAL_SMOOTH / "SUMMARY.json")
        for hist_bank in ("fg_lock_bg24", "bg_smooth", "dual_smooth", "fg48_bg12", "rankfocus"):
            if hist_bank in prev_sum.get("banks", {}) and hist_bank in prev_sum.get("visual_metrics", {}):
                shampoo_history[hist_bank] = dict(
                    metrics=prev_sum["banks"][hist_bank]["metrics"]["07_shampoo"],
                    visual=prev_sum["visual_metrics"][hist_bank]["07_shampoo"],
                )

    summary = dict(
        schema_version=1,
        status="completed",
        scope="retrieval_only",
        query_mode="joint",
        pool_count=189,
        simultaneous_replacements=len(GROUPS),
        query_count_per_group=20,
        total_queries_per_bank=len(GROUPS) * 20,
        shopping_executed=False,
        inputs_sha256=sha256_file(run / "EVALUATION_INPUTS.json"),
        verification_sha256=sha256_file(run / "INDEPENDENT_VERIFICATION.json"),
        visual_metrics=visual,
        shampoo_historical_reference=shampoo_history,
        banks={}
    )

    references = [b for b in BANKS if b != "ambient_coop"]
    for name in BANKS:
        all_rows = [dict(row, task_id=f"{key}:{row['task_id']}") for key in GROUPS for row in results[name][key]["rankings"]]
        sections = {key: results[name][key]["metrics"] for key in GROUPS}
        sections["overall"] = _metrics([r["source_rank"] for r in all_rows])
        comparisons = {}
        for reference in references:
            comparisons[reference] = {}
            for key in (*GROUPS, "overall"):
                previous = ([dict(row, task_id=f"{g}:{row['task_id']}") for g in GROUPS for row in results[reference][g]["rankings"]]
                            if key == "overall" else results[reference][key]["rankings"])
                current = all_rows if key == "overall" else results[name][key]["rankings"]
                before, after = _metrics([r["source_rank"] for r in previous]), sections[key]
                comparisons[reference][key] = dict(
                    **_paired(current, previous),
                    mean_rank_reduction=before["mean_rank"] - after["mean_rank"],
                    mrr_gain=after["mrr"] - before["mrr"],
                    **{f"top{k}_gain": after[f"top{k}"] - before[f"top{k}"] for k in (1, 3, 5)},
                    **{f"top{k}_rate_gain": after[f"top{k}_rate"] - before[f"top{k}_rate"] for k in (1, 3, 5)}
                )
        summary["banks"][name] = dict(metrics=sections, comparisons=comparisons)
    freeze_json(run / "SUMMARY.json", summary)

    labels = {
        "clean": "原图库 (clean)",
        "baseline": "原无像素感知方法 (PerLAGE_no_pixel 基线)",
        "lucid": "原始像素方法 (lucid)",
        "ambient_coop": "环境光背景协同＋主体微光影 (ambient_coop)",
        "dual_smooth": "上一轮双平滑协同 (dual_smooth)",
        "bg_smooth": "主体锁定＋背景平滑过渡 (bg_smooth)",
        "fg_lock_bg24": "主体锁定＋背景未平滑24级 (fg_lock_bg24)",
        "fg48_bg12": "全图微幅协同档 (fg48_bg12)",
        "rankfocus": "千步主体强攻档 (rankfocus)",
    }
    group_labels = {
        "07_shampoo": "第七组：洗发水 (07_shampoo)",
        "09_toothbrush": "第九组：牙刷 (09_toothbrush)",
        "overall": "两组总体汇总 (overall, 40 请求)",
    }

    lines = [
        "# 第七组洗发水与第九组牙刷“环境光背景协同调整＋主体微光影优化”（Ambient-Coop）实验报告",
        "",
        "本方案严格遵循以下设计：",
        "1. **商品主体**：直接锚定在冻结 `PerLAGE_no_pixel` 最终成图（`anchor_source='perlage_final'`），仅施加 `latent_levels=6` 级的大核高斯平滑（`sigma=5.0/9.0`）、低色度（`fg_chroma_keep=0.30`）微光影修饰，并严格保护印刷文字区零改动（`change[roi] == 0`），维持 `PerLAGE_no_pixel` 原本干净自然的表面质感；",
        "2. **背景区域**：不强制固定为 `255` 纯白，而在无截断可行区间 `[r_min, r_max]` 的中点附近同步联合优化全局环境底色（`ambient_global`）、`8x8` 低频环境光网格（`ambient_grid`）与先高斯平滑后缩放的 VAE 连续场（`background_levels=28` 级），彻底消除 `255` 半波截断带来的“纯白孤岛＋彩色斑块”以及主体轮廓紧邻处的“纯白光晕环”。",
        "",
        "两组完整图片（`07_shampoo`、`09_toothbrush`）全部生成并冻结后，在完整 189 条共享长期记忆池中进行同步双图替换图文检索评测（无防御，每组 20 条、共 40 条正式图文请求），并在独立子进程中完成全量 30,240 个候选分数复算验证；未执行下游购物。",
        "",
        "## 一、 检索排名结果汇总（第七组洗发水、第九组牙刷及两组总体）",
        "",
    ]

    for key in (*GROUPS, "overall"):
        lines += [
            f"### {group_labels[key]}",
            "",
            "| 图库 / 方案 | 请求数 | 排名第一 (Top-1) | 进入前三 (Top-3) | 进入前五 (Top-5) | 平均名次 (MeanRank)↓ | 平均倒数排名 (MRR)↑ |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for name in BANKS:
            metrics = summary["banks"][name]["metrics"][key]
            cells = [f"{metrics[f'top{k}']}/{metrics['query_count']}（{metrics[f'top{k}_rate']:.1%}）" for k in (1, 3, 5)]
            lines.append(f"| {labels[name]} | {metrics['query_count']} | {' | '.join(cells)} | {metrics['mean_rank']:.3f} | {metrics['mrr']:.5f} |")
        lines.append("")

    lines += [
        "## 二、 外观自然度与主体／背景分区改动统计",
        "",
        "| 组别 | 方案档位 | 步数（最佳步） | 结构相似度 (SSIM)↑ | 全图 MAE (级)↓ | 主体相对原图 MAE (级) | 主体相对基线最大/平均改动 | 背景相对原图 MAE (级) | 背景相对基线最大/平均改动 | 背景平均 RGB 偏移 (级) | 训练联合得分 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for key in GROUPS:
        for prof in ("baseline", *PROFILES):
            v = visual[prof][key]
            fg_mae_str = f"{v['fg_source_mae_levels']:.2f}" if v.get("fg_source_mae_levels") is not None else "—"
            bg_mae_str = f"{v['bg_source_mae_levels']:.2f}" if v.get("bg_source_mae_levels") is not None else "—"
            fg_anc_str = f"{v['fg_latent_max_levels']} / {v['fg_latent_mae_levels']:.2f} 级"
            bg_anc_str = f"{v['bg_latent_max_levels']} / {v['bg_latent_mae_levels']:.2f} 级"
            rgb_shift = v.get("bg_mean_rgb_shift_levels", [0.0, 0.0, 0.0])
            rgb_str = f"({rgb_shift[0]:+.1f}, {rgb_shift[1]:+.1f}, {rgb_shift[2]:+.1f})"
            lines.append(
                f"| {group_labels[key]} | {labels[prof]} | {v['steps']}（{v['best_step']}） | "
                f"{v['source_ssim']:.4f} | {v['source_mae_levels']:.2f} | {fg_mae_str} | "
                f"{fg_anc_str} | {bg_mae_str} | {bg_anc_str} | {rgb_str} | {v['joint_score']:.4f} |"
            )

    if shampoo_history:
        lines += [
            "",
            "## 三、 第七组洗发水跨轮次方案纵向对比（20 条正式图文请求）",
            "",
            "| 方案档位 | 排名第一 (Top-1) | 进入前三 (Top-3) | 进入前五 (Top-5) | 平均名次 (MeanRank)↓ | 结构相似度 (SSIM)↑ | 主体相对原图 MAE (级) | 背景相对原图 MAE (级) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        m_base = summary["banks"]["baseline"]["metrics"]["07_shampoo"]
        v_base = visual["baseline"]["07_shampoo"]
        lines.append(
            f"| {labels['baseline']} | {m_base['top1']}/20 | {m_base['top3']}/20 | {m_base['top5']}/20 | "
            f"{m_base['mean_rank']:.3f} | {v_base['source_ssim']:.4f} | {v_base['fg_source_mae_levels']:.2f} | {v_base['bg_source_mae_levels']:.2f} |"
        )
        for hist_bank in ("fg_lock_bg24", "bg_smooth", "dual_smooth", "fg48_bg12", "rankfocus"):
            if hist_bank in shampoo_history:
                mh = shampoo_history[hist_bank]["metrics"]
                vh = shampoo_history[hist_bank]["visual"]
                fg_s = f"{vh['fg_source_mae_levels']:.2f}" if vh.get("fg_source_mae_levels") is not None else "—"
                bg_s = f"{vh['bg_source_mae_levels']:.2f}" if vh.get("bg_source_mae_levels") is not None else "—"
                lines.append(
                    f"| {labels[hist_bank]} | {mh['top1']}/20 | {mh['top3']}/20 | {mh['top5']}/20 | "
                    f"{mh['mean_rank']:.3f} | {vh['source_ssim']:.4f} | {fg_s} | {bg_s} |"
                )
        m_amb = summary["banks"]["ambient_coop"]["metrics"]["07_shampoo"]
        v_amb = visual["ambient_coop"]["07_shampoo"]
        lines.append(
            f"| **{labels['ambient_coop']}** | **{m_amb['top1']}/20** | **{m_amb['top3']}/20** | **{m_amb['top5']}/20** | "
            f"**{m_amb['mean_rank']:.3f}** | **{v_amb['source_ssim']:.4f}** | **{v_amb['fg_source_mae_levels']:.2f}** | **{v_amb['bg_source_mae_levels']:.2f}** |"
        )

    lines += [
        "",
        "## 四、 与本轮对照基线配对差异（Ambient-Coop 表现增量）",
        "",
        "| 组别范围 | 对照基准 | 前三净增加数 | 前五净增加数 | 配对改善请求数 | 配对退化请求数 | 平均名次改善↓ | 倒数排名增量 (MRR)↑ |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key in (*GROUPS, "overall"):
        for reference in references:
            delta = summary["banks"]["ambient_coop"]["comparisons"][reference][key]
            lines.append(
                f"| {group_labels[key]} | {labels[reference]} | {delta['top3_gain']:+d} | {delta['top5_gain']:+d} | "
                f"{delta['improved_ranks']} | {delta['worsened_ranks']} | {delta['mean_rank_reduction']:+.3f} | {delta['mrr_gain']:+.5f} |"
            )

    lines += [
        "",
        "## 五、 独立子进程复算与审计结论",
        "",
        f"- **复算进程隔离**：主评测进程 PID `{verification['retrieval_pid']}`，独立复算进程 PID `{verification['pid']}`（严格进程隔离）。",
        f"- **全量候选比对**：重新计算全部 {verification['rankings_recomputed']} 个排名与 {verification['score_comparisons']} 个候选分数，逐项核验 ID、图片摘要与排序完全一致。",
        f"- **最大浮点误差**：`{verification['max_score_error']:.9g}`（严格满足 `<= 1e-6` 容差门限）。",
        "- **客观性保证**：本报告如实呈现预定生成与冻结图库排名，严禁基于正式检索结果重选图片或重训调参。",
        ""
    ]
    report_path = run / "REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return summary
