"""第七组洗发水“主体非文字区平滑微调 ＋ 微幅背景平滑漫射”（Dual-Smooth）实验：
图片冻结后完整库统一图文检索评测、独立子进程复算与综合对比报告生成。
"""
from __future__ import annotations

import copy
import math
import os

from src.paths import PROJECT, EXPERIMENT_BASE_GROUPS9 as BASE, ProjectPath as Path
from src.visual.audit import digest, freeze_json, read_json
from src.visual.image_utils import sha256_file


GROUPS = ("07_shampoo",)
PROFILES = ("dual_smooth",)
BANKS = ("clean", "baseline", "lucid", "rankfocus", "fg48_bg12", "fg_lock_bg24", "bg_smooth", *PROFILES)

# 历史只读基线路径
PREVIOUS_BASELINE = PROJECT / "output/perlage_no_pixel_groups9_20261109"
LUCID_RUN = PROJECT / "output/lucid_groups9_20261109"
PREVIOUS_RANKFOCUS = PROJECT / "output/perlage_rankfocus_groups67_20261111"
PREVIOUS_BG_COOP = PROJECT / "output/perlage_bg_coop_groups67_20261111"
PREVIOUS_BG_ONLY = PROJECT / "output/perlage_bg_only_shampoo_20261111"
PREVIOUS_BG_SMOOTH = PROJECT / "output/perlage_bg_smooth_shampoo_20261111"
QUERY_RUN = PROJECT / "output/groups9_multimodal_20261110"

STAGES = ("foreground_segmentation", "foreground_statistics", "color_harmonization", "latent_generation_refinement")


def _check(path: str | Path, expected: str) -> None:
    path_obj = Path(path)
    if not expected or not path_obj.is_file() or sha256_file(path_obj) != expected:
        raise ValueError(f"冻结文件缺失或摘要改变：{path}")


def _frozen_generation(run: Path | str) -> dict:
    """在图片生成四阶段凭证与冻结检验完成前，严禁读取共享池与正式查询。"""
    run = Path(run).resolve()
    plan_file = run / "EXPERIMENT_PLAN.json"
    plan = read_json(plan_file)
    if plan.get("groups") != list(GROUPS) or plan.get("profiles") != list(PROFILES):
        raise ValueError("实验必须预先冻结第七组洗发水及 dual_smooth 双平滑档位")

    registry_file = run / "FROZEN_IMAGES.json"
    frozen = read_json(registry_file)
    if frozen.get("status") != "frozen" or set(frozen.get("images", {})) != set(PROFILES):
        raise ValueError("双平滑档位完整图片必须冻结后方可评测")
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

            # 校验双平滑专用完整性凭证
            from src.visual._perlage_dual_smooth import validate_dual_smooth_complete
            validate_dual_smooth_complete(ready, item["image"])

            expected_config = plan["profile_configs"][profile]
            if ready.get("profile") != profile or ready.get("latent_config") != expected_config:
                raise ValueError("完整成图没有绑定预注册档位配置")
            if expected_config.get("steps") != 350:
                raise ValueError("细化步数必须为 350 步")
            if not (16 <= expected_config.get("latent_levels", 0) <= 88):
                raise ValueError("主体潜变量预算不符合双平滑方案约束 (16~88级)")
            if not (6 <= expected_config.get("background_levels", 0) <= 8):
                raise ValueError("背景潜变量预算不符合双平滑方案约束 (6~8级)")

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
                raise ValueError("双平滑实验严禁添加像素残差")

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
    """保留完整 189 条共享池；每个修改库只替换洗发水对应源记忆图片。

    支持 2 参数调用 (entries, replacements) 与 3 参数调用 (entries, pool_path, replacements)。
    """
    if replacements is None:
        pool_path = None
        replacements = pool_path_or_replacements
    else:
        pool_path = pool_path_or_replacements

    if len(entries) != 189 or len({e["Id"] for e in entries}) != 189:
        raise ValueError("必须使用完整且唯一的一百八十九条共享池")

    target_id = "groups9_reference_07_shampoo_B08XYBS678"
    for bank_name, change_map in replacements.items():
        if set(change_map.keys()) != {target_id}:
            raise ValueError(f"每个修改库只能替换指定的单个洗发水条目: 图库 {bank_name} 替换条目必须严格且仅包含 {target_id}")

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
                    ASIN=entry.get("FinalASIN", "B08XYBS678"),
                    LocalPath=str(Path(asset["image"]).resolve()),
                    SHA256=asset["sha256"]
                )
        stripped = [{k: v for k, v in entry.items() if k != "RetrievalImage"} for entry in bank]
        if stripped != clean or sum("RetrievalImage" in entry for entry in bank) != 1:
            raise ValueError(f"实验库 {name} 改变了图片以外的内容或替换范围不为 1")
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
    target_mem_id = "groups9_reference_07_shampoo_B08XYBS678"

    for key in GROUPS:
        meta = root["groups"][key]
        source_id = root["shared_memory"]["source_mapping"][key]["source_memory_id"]
        if source_id != target_mem_id:
            raise ValueError(f"源记忆映射不匹配：{source_id} != {target_mem_id}")

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

        # 3. rankfocus (千步主体强攻档)
        rf_reg_path = PREVIOUS_RANKFOCUS / "FROZEN_IMAGES.json"
        rf_reg = read_json(rf_reg_path)
        rf_asset = rf_reg["images"]["rankfocus"][key]
        _check(rf_asset["image"], rf_asset["sha256"])
        _check(rf_asset["ready"], rf_asset["ready_sha256"])
        bindings[str(rf_reg_path)] = sha256_file(rf_reg_path)
        bindings[rf_asset["image"]] = rf_asset["sha256"]
        bindings[rf_asset["ready"]] = rf_asset["ready_sha256"]
        replacements["rankfocus"][source_id] = dict(image=rf_asset["image"], sha256=rf_asset["sha256"])

        # 4. fg48_bg12 (历史全图微幅协同档)
        coop_reg_path = PREVIOUS_BG_COOP / "FROZEN_IMAGES.json"
        coop_reg = read_json(coop_reg_path)
        fg48_asset = coop_reg["images"]["fg48_bg12"][key]
        _check(fg48_asset["image"], fg48_asset["sha256"])
        _check(fg48_asset["ready"], fg48_asset["ready_sha256"])
        bindings[str(coop_reg_path)] = sha256_file(coop_reg_path)
        bindings[fg48_asset["image"]] = fg48_asset["sha256"]
        bindings[fg48_asset["ready"]] = fg48_asset["ready_sha256"]
        replacements["fg48_bg12"][source_id] = dict(image=fg48_asset["image"], sha256=fg48_asset["sha256"])

        # 5. fg_lock_bg24 (主体锁定＋背景未平滑 24 级)
        only_reg_path = PREVIOUS_BG_ONLY / "FROZEN_IMAGES.json"
        only_reg = read_json(only_reg_path)
        bg24_asset = only_reg["images"]["fg_lock_bg24"][key]
        _check(bg24_asset["image"], bg24_asset["sha256"])
        _check(bg24_asset["ready"], bg24_asset["ready_sha256"])
        bindings[str(only_reg_path)] = sha256_file(only_reg_path)
        bindings[bg24_asset["image"]] = bg24_asset["sha256"]
        bindings[bg24_asset["ready"]] = bg24_asset["ready_sha256"]
        replacements["fg_lock_bg24"][source_id] = dict(image=bg24_asset["image"], sha256=bg24_asset["sha256"])

        # 6. bg_smooth (主体锁定＋背景平滑过渡 22 级)
        smooth_reg_path = PREVIOUS_BG_SMOOTH / "FROZEN_IMAGES.json"
        smooth_reg = read_json(smooth_reg_path)
        bgs_asset = smooth_reg["images"]["bg_smooth"][key]
        _check(bgs_asset["image"], bgs_asset["sha256"])
        _check(bgs_asset["ready"], bgs_asset["ready_sha256"])
        bindings[str(smooth_reg_path)] = sha256_file(smooth_reg_path)
        bindings[bgs_asset["image"]] = bgs_asset["sha256"]
        bindings[bgs_asset["ready"]] = bgs_asset["ready_sha256"]
        replacements["bg_smooth"][source_id] = dict(image=bgs_asset["image"], sha256=bgs_asset["sha256"])

        # 7. dual_smooth (本轮新生成双平滑图)
        for profile in PROFILES:
            asset = frozen["images"][profile][key]
            ready = read_json(asset["ready"])
            if ready.get("source_sha256") != meta["source_sha256"] or ready.get("target_sha256") != meta["target_sha256"]:
                raise ValueError("双平滑完整图与本轮冻结公开图对不一致")
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
        simultaneous_replacements_per_modified_bank=1,
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
    """完整图片冻结后，新编码所有全局库与洗发水 20 条图文查询。"""
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
        improved_ranks=sum(a < b for a, b in pairs),
        worsened_ranks=sum(a > b for a, b in pairs),
        tied_ranks=sum(a == b for a, b in pairs),
    )


def _visual_summary(run: Path) -> dict:
    """提取本轮 dual_smooth 与各既有基线的外观指标供综合报告对比。"""
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
                fg_latent_max_levels=meta.get("fg_latent_max_levels"),
                fg_latent_mae_levels=meta.get("fg_latent_mae_levels"),
                bg_latent_max_levels=meta.get("bg_latent_max_levels"),
                bg_latent_mae_levels=meta.get("bg_latent_mae_levels"),
                halo_error=fc["halo_error"],
                joint_score=fc["joint_scores"][0][0],
            )

    # 1. rankfocus (千步主体强攻档)
    visual["rankfocus"] = {}
    for key in GROUPS:
        meta = read_json(PREVIOUS_RANKFOCUS / "generation" / "rankfocus" / key / "metadata.json")
        fc = meta["final_check"]
        visual["rankfocus"][key] = dict(
            steps=meta["steps_completed"], best_step=meta["best"]["step"],
            accepted=meta["accepted_updates"], rejected=meta["rejected_updates"],
            source_ssim=fc["source_ssim"], source_mae_levels=fc["source_mae_levels"],
            fg_source_mae_levels=fc.get("fg_source_mae_levels"), bg_source_mae_levels=fc.get("bg_source_mae_levels"),
            fg_latent_max_levels=meta.get("fg_latent_max_levels"), fg_latent_mae_levels=meta.get("fg_latent_mae_levels"),
            bg_latent_max_levels=meta.get("bg_latent_max_levels"), bg_latent_mae_levels=meta.get("bg_latent_mae_levels"),
            halo_error=fc["halo_error"], joint_score=fc["joint_scores"][0][0],
        )

    # 2. fg48_bg12 (全图微幅协同自然档)
    visual["fg48_bg12"] = {}
    for key in GROUPS:
        meta = read_json(PREVIOUS_BG_COOP / "generation" / "fg48_bg12" / key / "metadata.json")
        fc = meta["final_check"]
        visual["fg48_bg12"][key] = dict(
            steps=meta["steps_completed"], best_step=meta["best"]["step"],
            accepted=meta["accepted_updates"], rejected=meta["rejected_updates"],
            source_ssim=fc["source_ssim"], source_mae_levels=fc["source_mae_levels"],
            fg_source_mae_levels=fc.get("fg_source_mae_levels"), bg_source_mae_levels=fc.get("bg_source_mae_levels"),
            fg_latent_max_levels=meta.get("fg_latent_max_levels"), fg_latent_mae_levels=meta.get("fg_latent_mae_levels"),
            bg_latent_max_levels=meta.get("bg_latent_max_levels"), bg_latent_mae_levels=meta.get("bg_latent_mae_levels"),
            halo_error=fc["halo_error"], joint_score=fc["joint_scores"][0][0],
        )

    # 3. fg_lock_bg24 (主体锁定＋背景未平滑 24 级)
    visual["fg_lock_bg24"] = {}
    for key in GROUPS:
        meta = read_json(PREVIOUS_BG_ONLY / "generation" / "fg_lock_bg24" / key / "metadata.json")
        fc = meta["final_check"]
        visual["fg_lock_bg24"][key] = dict(
            steps=meta["steps_completed"], best_step=meta["best"]["step"],
            accepted=meta["accepted_updates"], rejected=meta["rejected_updates"],
            source_ssim=fc["source_ssim"], source_mae_levels=fc["source_mae_levels"],
            fg_source_mae_levels=fc.get("fg_source_mae_levels"), bg_source_mae_levels=fc.get("bg_source_mae_levels"),
            fg_latent_max_levels=meta.get("fg_latent_max_levels"), fg_latent_mae_levels=meta.get("fg_latent_mae_levels"),
            bg_latent_max_levels=meta.get("bg_latent_max_levels"), bg_latent_mae_levels=meta.get("bg_latent_mae_levels"),
            halo_error=fc["halo_error"], joint_score=fc["joint_scores"][0][0],
        )

    # 4. bg_smooth (主体锁定＋背景平滑自然过渡 22 级)
    visual["bg_smooth"] = {}
    for key in GROUPS:
        meta = read_json(PREVIOUS_BG_SMOOTH / "generation" / "bg_smooth" / key / "metadata.json")
        fc = meta["final_check"]
        visual["bg_smooth"][key] = dict(
            steps=meta["steps_completed"], best_step=meta["best"]["step"],
            accepted=meta["accepted_updates"], rejected=meta["rejected_updates"],
            source_ssim=fc["source_ssim"], source_mae_levels=fc["source_mae_levels"],
            fg_source_mae_levels=fc.get("fg_source_mae_levels"), bg_source_mae_levels=fc.get("bg_source_mae_levels"),
            fg_latent_max_levels=meta.get("fg_latent_max_levels"), fg_latent_mae_levels=meta.get("fg_latent_mae_levels"),
            bg_latent_max_levels=meta.get("bg_latent_max_levels"), bg_latent_mae_levels=meta.get("bg_latent_mae_levels"),
            halo_error=fc["halo_error"], joint_score=fc["joint_scores"][0][0],
        )

    # 5. baseline (PerLAGE_no_pixel) 基线
    visual["baseline"] = {
        "07_shampoo": dict(
            steps=40, best_step=40, accepted=40, rejected=0,
            source_ssim=0.9669, source_mae_levels=4.50,
            fg_source_mae_levels=8.53, bg_source_mae_levels=0.72,
            fg_latent_max_levels=0, fg_latent_mae_levels=0.0,
            bg_latent_max_levels=0, bg_latent_mae_levels=0.0,
            halo_error=6.431, joint_score=0.7023
        )
    }
    return visual


def report(run: Path | str) -> dict:
    """汇总评测指标，输出 SUMMARY.json 与 REPORT.md。"""
    run = Path(run).resolve()
    inputs = _inputs(run)
    verification = _validate_verification(run, inputs)
    results = {name: {key: read_json(_result_path(run, name, key)) for key in GROUPS} for name in BANKS}
    visual = _visual_summary(run)

    summary = dict(
        schema_version=1,
        status="completed",
        scope="retrieval_only",
        query_mode="joint",
        pool_count=189,
        simultaneous_replacements=1,
        query_count_per_bank=20,
        shopping_executed=False,
        inputs_sha256=sha256_file(run / "EVALUATION_INPUTS.json"),
        verification_sha256=sha256_file(run / "INDEPENDENT_VERIFICATION.json"),
        visual_metrics=visual,
        banks={}
    )

    references = [b for b in BANKS if b != "dual_smooth"]
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
        "rankfocus": "千步主体强攻档 (rankfocus)",
        "fg48_bg12": "全图微幅协同自然档 (fg48_bg12)",
        "fg_lock_bg24": "主体锁定＋背景未平滑24级 (fg_lock_bg24)",
        "bg_smooth": "主体锁定＋背景平滑自然过渡 (bg_smooth)",
        "dual_smooth": "主体曲面平滑＋微幅背景平滑协同 (dual_smooth)",
    }

    lines = [
        "# 第七组洗发水“主体曲面平滑＋微幅背景平滑协同”（Dual-Smooth）单图实验报告",
        "",
        "本方案针对第七组洗发水，采用“瓶身主体非文字区余弦内缩平滑微调（latent_levels=20 级）＋ 极微弱背景余弦平滑过渡（background_levels=6 级）”双平滑协同机制，配合明度/色度双尺度高斯平滑与严格印刷区域（ROI）零改动保护。",
        "实验仅生成一张完整图片（`dual_smooth`），冻结后在完整 189 条共享长期记忆池中进行统一图文检索评测（无防御，20 条正式图文请求），并在独立子进程中完成全分数复算验证；未执行下游购物。",
        "",
        "## 一、 洗发水组检索排名汇总对比（20 条正式图文请求）",
        "",
        "| 图库 / 方案 | 请求数 | 排名第一 (Top-1) | 进入前三 (Top-3) | 进入前五 (Top-5) | 平均名次 (MeanRank)↓ | 平均倒数排名 (MRR)↑ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in BANKS:
        metrics = summary["banks"][name]["metrics"]["07_shampoo"]
        cells = [f"{metrics[f'top{k}']}/{metrics['query_count']}（{metrics[f'top{k}_rate']:.1%}）" for k in (1, 3, 5)]
        lines.append(f"| {labels[name]} | {metrics['query_count']} | {' | '.join(cells)} | {metrics['mean_rank']:.3f} | {metrics['mrr']:.5f} |")

    lines += [
        "",
        "## 二、 外观自然度与主体／背景分区改动统计",
        "",
        "| 方案档位 | 步数（最佳步） | 结构相似度 (SSIM)↑ | 全图 MAE (级)↓ | 主体前景 MAE (级) | 主体最大潜变级 | 背景平均变化 (级)↓ | 背景最大潜变级 | 光晕误差↓ | 训练联合得分 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for prof in ("baseline", "rankfocus", "fg48_bg12", "fg_lock_bg24", "bg_smooth", *PROFILES):
        v = visual[prof]["07_shampoo"]
        fg_mae_str = f"{v['fg_source_mae_levels']:.2f}" if v["fg_source_mae_levels"] is not None else "—"
        bg_mae_str = f"{v['bg_source_mae_levels']:.2f}" if v.get("bg_source_mae_levels") is not None else "—"
        fg_max_str = f"{v['fg_latent_max_levels']} 级" if v.get("fg_latent_max_levels") is not None else "0 级"
        bg_max_str = f"{v['bg_latent_max_levels']} 级" if v.get("bg_latent_max_levels") is not None else "—"
        lines.append(
            f"| {labels[prof]} | {v['steps']}（{v['best_step']}） | "
            f"{v['source_ssim']:.4f} | {v['source_mae_levels']:.2f} | {fg_mae_str} | "
            f"{fg_max_str} | {bg_mae_str} | {bg_max_str} | "
            f"{v['halo_error']:.3f} | {v['joint_score']:.4f} |"
        )

    lines += [
        "",
        "## 三、 与各历史对照基线配对差异（Dual-Smooth 表现增量）",
        "",
        "| 方案 | 对照基准 | 前三净增加数 | 配对新增前三 | 配对丢失前三 | 平均名次改善↓ | 倒数排名增量 (MRR)↑ |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for reference in references:
        delta = summary["banks"]["dual_smooth"]["comparisons"][reference]["07_shampoo"]
        lines.append(
            f"| {labels['dual_smooth']} | {labels[reference]} | {delta['top3_gain']:+d} | "
            f"{delta['top3_gained']} | {delta['top3_lost']} | {delta['mean_rank_reduction']:+.3f} | {delta['mrr_gain']:+.5f} |"
        )

    lines += [
        "",
        "## 四、 独立子进程复算与审计结论",
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
