"""单组分布式无像素实验：三图冻结后七个完整库统一对照及独立全分数复算。"""
from __future__ import annotations

import copy
import math
import os

from src.paths import PROJECT, EXPERIMENT_BASE_GROUPS9 as BASE, ProjectPath as Path
from src.visual.audit import digest, freeze_json, read_json
from src.visual.image_utils import sha256_file


GROUPS = ("07_shampoo",)
PROFILES = ("foreground_control", "distributed_balanced", "distributed_flexible")
BANKS = ("clean", "baseline", "lucid", "rankfocus", *PROFILES)
PREVIOUS = PROJECT / "output/perlage_no_pixel_groups9_20261109"
QUERY_RUN = PROJECT / "output/groups9_multimodal_20261110"
LUCID_RUN = PROJECT / "output/lucid_groups9_20261109"
RANKFOCUS_RUN = PROJECT / "output/perlage_rankfocus_groups67_20261111"
STAGES = ("foreground_segmentation", "foreground_statistics", "color_harmonization", "latent_generation_refinement")


def _check(path, expected):
    if not expected or not Path(path).is_file() or sha256_file(path) != expected:
        raise ValueError("冻结文件缺失或摘要改变：" + str(path))


def _frozen_generation(run):
    """此检查完成前不读共享池、正式查询或任何检索结果。"""
    run = Path(run).resolve()
    plan_file = run / "EXPERIMENT_PLAN.json"
    plan = read_json(plan_file)
    if plan.get("groups") != list(GROUPS) or plan.get("profiles") != list(PROFILES):
        raise ValueError("实验必须预先冻结仅第七组及三个分布式对照档位")
    registry_file = run / "FROZEN_IMAGES.json"
    frozen = read_json(registry_file)
    if frozen.get("status") != "frozen" or set(frozen.get("images", {})) != set(PROFILES):
        raise ValueError("三个档位三张完整图片必须全部冻结后评估")
    if frozen.get("plan_sha256") != sha256_file(plan_file):
        raise ValueError("冻结图片未绑定预定训练计划")
    for profile in PROFILES:
        if set(frozen["images"][profile]) != set(GROUPS):
            raise ValueError("冻结图必须只包含第七组")
        for key in GROUPS:
            item = frozen["images"][profile][key]
            _check(item["image"], item["sha256"])
            _check(item["ready"], item["ready_sha256"])
            ready = read_json(item["ready"])
            from src.visual._perlage_no_pixel_distributed import validate_distributed_complete
            validate_distributed_complete(ready, item["image"])
            expected_config = plan["profile_configs"][profile]
            if (ready.get("profile") != profile or ready.get("latent_config") != expected_config
                    or expected_config.get("steps") != 1000):
                raise ValueError("完整成图没有绑定预注册档位及最大一千步配置")
            public_data = plan["generation_inputs"][key]["definition"]["data"]
            for role in ("source", "target"):
                if ready.get(role + "_sha256") != sha256_file(public_data[role]):
                    raise ValueError("完整成图与预注册公开图对不一致：" + role)
            if ready.get("status") != "completed" or ready.get("pixel_optimization") is not False:
                raise ValueError("放宽实验要求完整无像素成图")
            if any(ready.get(field) is not False for field in
                   ("formal_queries_used", "victim_memory_used", "shadow_pool_used", "purchase_history_used")):
                raise ValueError("生成证据未明确排除正式查询、记忆和影子库训练")
            if ready.get("auxiliary_max_levels", 0) != 0 or ready.get("latent_config", {}).get("auxiliary_levels", 0) != 0:
                raise ValueError("放宽实验不得添加像素残差")
            stages = ready.get("stages", {})
            if set(stages) != set(STAGES):
                raise ValueError("无像素成图须保留完整四阶段")
            for stage in STAGES:
                record = stages[stage]
                if record.get("status") != "completed" or not record.get("files"):
                    raise ValueError("生成阶段证据不完整：" + stage)
                for path, expected in record["files"].items():
                    _check(path, expected)
            latent = stages["latent_generation_refinement"]
            stop = ready.get("early_stop", {})
            if (type(latent.get("steps_completed")) is not int
                    or not 300 <= latent["steps_completed"] <= 1000
                    or latent.get("steps_requested") != 1000
                    or latent.get("schedule") != "adaptive_public_plateau_v1"
                    or stop.get("steps_completed") != latent["steps_completed"]
                    or stop.get("rule_satisfied") is not True):
                raise ValueError("最终潜变量细化没有完成预注册自适应停止日程")
            final = ready.get("candidates", {}).get("final", {})
            if final.get("sha256") != item["sha256"]:
                raise ValueError("冻结图与完整方法最终候选不是同一图片")
    return frozen


def build_banks(entries, pool_path, replacements):
    """保留完整共享池；每个修改库只替换相同一条图片。"""
    from src.visual.visual_retrieval import resolve_image
    if len(entries) != 189 or len({e["Id"] for e in entries}) != 189:
        raise ValueError("必须使用完整且唯一的一百八十九条共享池")
    if set(replacements) != set(BANKS) - {"clean"}:
        raise ValueError("替换清单必须包括三种既有方法和三个新档位")
    ids = set(replacements["baseline"])
    if len(ids) != 1 or any(set(row) != ids for row in replacements.values()):
        raise ValueError("六个修改库必须只替换相同的一条源记忆")
    if not ids <= {entry["Id"] for entry in entries}:
        raise ValueError("替换条目不在完整共享池")
    clean = copy.deepcopy(entries)
    for entry in clean:
        if entry.get("RetrievalImage") or entry.get("ImageSubstitution"):
            raise ValueError("基础共享池不能含旧图片替换")
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
                _check(asset["image"], asset["sha256"])
                entry["RetrievalImage"] = dict(ASIN=entry["FinalASIN"], LocalPath=str(Path(asset["image"]).resolve()),
                                               SHA256=asset["sha256"])
        stripped = [{k: v for k, v in entry.items() if k != "RetrievalImage"} for entry in bank]
        if stripped != clean or sum("RetrievalImage" in entry for entry in bank) != 1:
            raise ValueError("实验库改变了图片以外的内容或替换范围")
        banks[name] = bank
    return banks


def _prepare_inputs(run):
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
    bindings = {str(BASE / "manifest.json"): sha256_file(BASE / "manifest.json"),
                str(pool_path): sha256_file(pool_path), str(encoder_path): sha256_file(encoder_path),
                str(run / "EXPERIMENT_PLAN.json"): sha256_file(run / "EXPERIMENT_PLAN.json"),
                str(run / "FROZEN_IMAGES.json"): sha256_file(run / "FROZEN_IMAGES.json")}
    groups, replacements = {}, {name: {} for name in BANKS if name != "clean"}
    for key in GROUPS:
        meta = root["groups"][key]
        source_id = root["shared_memory"]["source_mapping"][key]["source_memory_id"]
        original_file = PREVIOUS / "groups" / key / "methods/PerLAGE_no_pixel/FROZEN_IMAGE.json"
        original = read_json(original_file)
        if (original.get("status") != "frozen" or original["source"]["id"] != meta["source_id"]
                or original["target"]["id"] != meta["target_id"]):
            raise ValueError("原无像素完整图的商品身份不符")
        _check(original["image"], original["image_sha256"])
        for role in ("source", "target"):
            _check(original[role]["image"], original[role]["image_sha256"])
            if original[role]["image_sha256"] != meta[role + "_sha256"]:
                raise ValueError("原无像素对照与本轮公开图对不一致")
        bindings[str(original_file)] = sha256_file(original_file)
        bindings[original["image"]] = original["image_sha256"]
        replacements["baseline"][source_id] = dict(image=original["image"], sha256=original["image_sha256"])
        lucid_file = LUCID_RUN / "groups" / key / "methods/lucid/FROZEN_IMAGE.json"
        lucid = read_json(lucid_file)
        if (lucid.get("status") != "frozen" or lucid["source"]["id"] != meta["source_id"]
                or lucid["target"]["id"] != meta["target_id"]):
            raise ValueError("原始像素方法完整图的商品身份不符")
        for role in ("source", "target"):
            if lucid[role]["image_sha256"] != meta[role + "_sha256"]:
                raise ValueError("原始像素方法没有使用同一公开图对")
        _check(lucid["image"], lucid["image_sha256"])
        bindings[str(lucid_file)] = sha256_file(lucid_file)
        bindings[lucid["image"]] = lucid["image_sha256"]
        replacements["lucid"][source_id] = dict(image=lucid["image"], sha256=lucid["image_sha256"])
        old_registry_path = RANKFOCUS_RUN / "FROZEN_IMAGES.json"
        old_registry = read_json(old_registry_path)
        if old_registry.get("status") != "frozen":
            raise ValueError("上一轮高收益强攻图尚未冻结")
        previous_asset = old_registry["images"]["rankfocus"][key]
        _check(previous_asset["image"], previous_asset["sha256"])
        _check(previous_asset["ready"], previous_asset["ready_sha256"])
        previous_ready = read_json(previous_asset["ready"])
        from src.visual._perlage_no_pixel_rankfocus import validate_rankfocus_complete
        validate_rankfocus_complete(previous_ready, previous_asset["image"])
        if (previous_ready.get("source_sha256") != meta["source_sha256"]
                or previous_ready.get("target_sha256") != meta["target_sha256"]
                or previous_ready.get("profile") != "rankfocus"):
            raise ValueError("上一轮高收益强攻图的来源或档位不符")
        bindings[str(old_registry_path)] = sha256_file(old_registry_path)
        bindings[previous_asset["image"]] = previous_asset["sha256"]
        bindings[previous_asset["ready"]] = previous_asset["ready_sha256"]
        replacements["rankfocus"][source_id] = dict(image=previous_asset["image"], sha256=previous_asset["sha256"])
        for profile in PROFILES:
            asset = frozen["images"][profile][key]
            ready = read_json(asset["ready"])
            if ready.get("source_sha256") != meta["source_sha256"] or ready.get("target_sha256") != meta["target_sha256"]:
                raise ValueError("放宽实验完整图与本轮冻结图对不一致")
            replacements[profile][source_id] = dict(image=asset["image"], sha256=asset["sha256"])
            bindings[asset["image"]] = asset["sha256"]
            bindings[asset["ready"]] = asset["ready_sha256"]
            metadata_file = Path(asset["ready"]).with_name("metadata.json")
            bindings[str(metadata_file)] = sha256_file(metadata_file)
        manifest_file = QUERY_RUN / "input_resources/formal_queries" / key / "queries.json"
        query_document = read_json(manifest_file)
        text_file = BASE / meta["query_manifest"]
        _check(text_file, meta["query_manifest_sha256"])
        text_doc = read_json(text_file)
        if query_document.get("instructions") != text_doc["instructions"]:
            raise ValueError("图文检索必须逐字复用既有特点文字")
        queries = load_formal_queries(manifest_file, original["target"], query_mode="joint")
        if len(queries) != 20:
            raise ValueError("每组必须保留二十条正式图文请求")
        bindings[str(manifest_file)] = sha256_file(manifest_file)
        bindings[str(text_file)] = sha256_file(text_file)
        for query in queries:
            bindings[query["query_image"]] = query["query_image_sha256"]
        groups[key] = dict(name=meta["name"], source_id=meta["source_id"], target_id=meta["target_id"],
                           source_memory_id=source_id, queries=queries, training_overlap=query_document.get("training_overlap"))
    banks = build_banks(entries, pool_path, replacements)
    paths = {}
    for name in BANKS:
        path = run / "banks" / (name + ".json")
        freeze_json(path, banks[name])
        paths[name] = dict(path=str(path), sha256=sha256_file(path))
        bindings[str(path)] = sha256_file(path)
    inputs = dict(schema_version=1, groups=groups, bank_order=list(BANKS), pool_count=189,
                  banks=paths, encoder_config=str(encoder_path), encoder_config_sha256=sha256_file(encoder_path),
                  bindings=bindings, query_mode="joint", query_image_weight=0.5, key_image_weight=0.5,
                  index_text=detect_index_text_policy(entries), condition="none", shopping_allowed=False,
                  simultaneous_replacements_per_modified_bank=1, expected_rankings=140, expected_score_comparisons=26460,
                  baseline_policy="recompute_original_perlage_images_for_both_groups_in_same_complete_pool",
                  current_round_rankings_used_for_selection=False,
                  historical_group_results_seen_for_development=True,
                  input_read_only_after_all_three_complete_images=True)
    freeze_json(run / "EVALUATION_INPUTS.json", inputs)
    return inputs


def _inputs(run):
    _frozen_generation(run)
    inputs = read_json(run / "EVALUATION_INPUTS.json")
    if (inputs.get("bank_order") != list(BANKS) or set(inputs.get("groups", {})) != set(GROUPS)
            or inputs.get("pool_count") != 189 or inputs.get("shopping_allowed") is not False):
        raise ValueError("冻结评估范围不符")
    for path, expected in inputs["bindings"].items():
        _check(path, expected)
    return inputs


class _MemoizedEncoder:
    """只复用当前独立进程本次新编码；不加载任何磁盘向量缓存。"""
    supports_image_weight = True

    def __init__(self, encoder):
        self.encoder, self.snapshot = encoder, encoder.snapshot
        self.cache, self.last_encoding_audit = {}, None

    def encode_joint(self, text, image, image_weight=0.5):
        key = (text, sha256_file(image), image_weight)
        if key not in self.cache:
            vector = self.encoder.encode_joint(text, image, image_weight=image_weight)
            self.cache[key] = (vector, copy.deepcopy(getattr(self.encoder, "last_encoding_audit", None)))
        vector, audit = self.cache[key]
        self.last_encoding_audit = copy.deepcopy(audit)
        return vector


def _ranker(encoder, run, inputs, name):
    from src.visual.visual_retrieval import VisualMemoryRetriever
    bank = read_json(inputs["banks"][name]["path"])
    if len(bank) != 189:
        raise ValueError("检索库未覆盖全部一百八十九条")
    ranker = VisualMemoryRetriever(encoder, inputs["banks"][name]["path"], cache_dir=None,
                                   query_mode="joint", key_image_weight=0.5, query_image_weight=0.5,
                                   index_text=inputs["index_text"])
    return bank, ranker


def _metrics(ranks):
    count = len(ranks)
    if not count or any(type(rank) is not int or not 1 <= rank <= 189 for rank in ranks):
        raise ValueError("正式排名范围无效")
    return dict(query_count=count, mean_rank=sum(ranks) / count, mrr=sum(1.0 / r for r in ranks) / count,
                **{f"top{k}": sum(r <= k for r in ranks) for k in (1, 3, 5)},
                **{f"top{k}_rate": sum(r <= k for r in ranks) / count for k in (1, 3, 5)})


def _result_path(run, name, key):
    return run / "retrieval" / name / (key + ".json")


def _validate_result(run, inputs, name, key):
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


def _validate_retrieval(run, inputs):
    certificate = read_json(run / "RETRIEVAL_COMPLETED.json")
    expected_files = {str(_result_path(run, n, k)) for n in BANKS for k in GROUPS}
    if (certificate.get("status") != "completed" or certificate.get("rankings") != 140
            or certificate.get("score_count") != 26460 or certificate.get("execution_device") != "cuda"
            or certificate.get("inputs_sha256") != sha256_file(run / "EVALUATION_INPUTS.json")
            or set(certificate.get("files", {})) != expected_files):
        raise ValueError("检索完成证书范围或输入绑定不完整")
    for path, expected in certificate["files"].items():
        _check(path, expected)
    for name in BANKS:
        for key in GROUPS:
            _validate_result(run, inputs, name, key)
    return certificate


def _validate_verification(run, inputs):
    retrieval = _validate_retrieval(run, inputs)
    verification = read_json(run / "INDEPENDENT_VERIFICATION.json")
    if (verification.get("status") != "passed" or verification.get("score_comparisons") != 26460
            or verification.get("rankings_recomputed") != 140 or verification.get("fresh_process") is not True
            or verification.get("fresh_encoder") is not True or verification.get("execution_device") != "cuda"
            or verification.get("pid") == retrieval.get("pid")
            or verification.get("retrieval_pid") != retrieval.get("pid")
            or verification.get("files") != retrieval.get("files")
            or verification.get("inputs_sha256") != sha256_file(run / "EVALUATION_INPUTS.json")
            or not isinstance(verification.get("max_score_error"), (int, float))
            or not math.isfinite(verification["max_score_error"]) or verification["max_score_error"] > 1e-6):
        raise ValueError("独立复算证书未绑定本轮全部候选分数")
    _check(run / "RETRIEVAL_COMPLETED.json", verification["retrieval_certificate_sha256"])
    return verification


def evaluate(run):
    """三图全冻结后，新编码七个全局库和同样二十条图文查询。"""
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
                    rows.append(dict(task_id=query["task_id"], source_rank=source["rank"], source_score=source["score"],
                                     ranking=ranking))
                freeze_json(path, dict(bank=name, group=key, protocol=copy.deepcopy(ranker.snapshot), rankings=rows,
                                       metrics=_metrics([row["source_rank"] for row in rows]),
                                       inputs_sha256=sha256_file(run / "EVALUATION_INPUTS.json")))
                print(f"检索完成：{name} {key}，二十请求、完整一百八十九条库", flush=True)
    finally:
        raw.close()
    files = {str(_result_path(run, n, k)): sha256_file(_result_path(run, n, k)) for n in BANKS for k in GROUPS}
    freeze_json(run / "RETRIEVAL_COMPLETED.json", dict(status="completed", pid=os.getpid(), execution_device="cuda",
                inputs_sha256=sha256_file(run / "EVALUATION_INPUTS.json"), rankings=140, score_count=26460,
                persistent_vector_cache_read=False, files=files))


def verify(run):
    """须在新进程执行；从零编码全部候选，比较每个请求的一百八十九项。"""
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
                            raise ValueError("独立复算候选分数差异超过百万分之一")
                        compared, maximum = compared + 1, max(maximum, error)
                    source = next(r for r in actual if r["Id"] == inputs["groups"][key]["source_memory_id"])
                    if source["rank"] != previous["source_rank"] or abs(source["score"] - previous["source_score"]) > 1e-6:
                        raise ValueError("保存的源记忆名次或分数与完整排名不符")
                    rankings += 1
                if old["metrics"] != _metrics([r["source_rank"] for r in old["rankings"]]):
                    raise ValueError("原始排名与汇总指标不一致")
    finally:
        raw.close()
    if compared != 26460 or rankings != 140:
        raise ValueError("独立复算没有覆盖全部一百四十排名及二万六千四百六十分数")
    freeze_json(run / "INDEPENDENT_VERIFICATION.json", dict(status="passed", fresh_process=True, pid=os.getpid(),
                retrieval_pid=retrieval["pid"], fresh_encoder=True, execution_device="cuda", rankings_recomputed=rankings,
                score_comparisons=compared, max_score_error=maximum, persistent_vector_cache_read=False,
                inputs_sha256=sha256_file(run / "EVALUATION_INPUTS.json"),
                retrieval_certificate_sha256=sha256_file(run / "RETRIEVAL_COMPLETED.json"), files=retrieval["files"]))


def _paired(current, previous):
    if [r["task_id"] for r in current] != [r["task_id"] for r in previous]:
        raise ValueError("配对名次的正式请求不一致")
    pairs = [(a["source_rank"], b["source_rank"]) for a, b in zip(current, previous)]
    return dict(top3_gained=sum(a <= 3 < b for a, b in pairs), top3_lost=sum(b <= 3 < a for a, b in pairs),
                improved_ranks=sum(a < b for a, b in pairs), worsened_ranks=sum(a > b for a, b in pairs),
                tied_ranks=sum(a == b for a, b in pairs))


def _maximum_patch_mae(error, region=None, size=32):
    """逐像素滑动的最差局部均值；区域块覆盖率至少八成且按区域归一化。"""
    import numpy as np
    pixel_error = np.asarray(error, dtype=np.float64).mean(axis=2)
    height, width = pixel_error.shape
    if height < size or width < size:
        return None
    mask = np.ones((height, width), dtype=np.float64) if region is None else np.asarray(region, dtype=np.float64)
    def windows(array):
        integral = np.pad(array, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        return (integral[size:, size:] - integral[:-size, size:]
                - integral[size:, :-size] + integral[:-size, :-size])
    counts = windows(mask)
    eligible = counts >= (size * size * (0.8 if region is not None else 1.0) - 1e-9)
    if not eligible.any():
        return None
    values = np.where(eligible, windows(pixel_error * mask) / np.maximum(counts, 1), -np.inf)
    y, x = np.unravel_index(np.argmax(values), values.shape)
    return dict(size=size, stride=1, mae_levels=float(values[y, x]),
                position_xy=[int(x), int(y)], box_xyxy=[int(x), int(y), int(x+size), int(y+size)],
                region_fraction=float(counts[y, x] / (size * size)),
                aggregation="regional_pixels_with_minimum_80_percent_coverage" if region is not None else "whole_patch")


def _quality_records(run):
    """仅统计冻结图片外观，不调用模型；分区域均值避免大片背景稀释。"""
    import numpy as np
    from PIL import Image
    plan = read_json(run / "EXPERIMENT_PLAN.json")
    registry = read_json(run / "FROZEN_IMAGES.json")
    previous = read_json(RANKFOCUS_RUN / "FROZEN_IMAGES.json")
    records, bindings = {}, {}
    for name in ("rankfocus", *PROFILES):
        records[name] = {}
        for key in GROUPS:
            asset = (previous["images"]["rankfocus"][key] if name == "rankfocus"
                     else registry["images"][name][key])
            metadata_path = Path(asset["ready"]).with_name("metadata.json")
            metadata = read_json(metadata_path)
            ready = read_json(asset["ready"])
            source_path = plan["generation_inputs"][key]["definition"]["data"]["source"]
            prefix = Path(plan["generation_inputs"][key]["prefix_root"])
            foreground_path = prefix / "foreground_segmentation/source/foreground.png"
            anchor_path = prefix / "color_harmonization/generative_anchor.png"
            image = np.asarray(Image.open(asset["image"]).convert("RGB"), dtype=np.float64)
            source = np.asarray(Image.open(source_path).convert("RGB"), dtype=np.float64)
            anchor = np.asarray(Image.open(anchor_path).convert("RGB"), dtype=np.float64)
            foreground = np.asarray(Image.open(foreground_path).convert("L")) > 0
            if image.shape != source.shape or anchor.shape != source.shape or not foreground.any() or foreground.all():
                raise ValueError("冻结外观统计图像或前景布局无效")
            difference = np.abs(image - source)
            addition = np.abs(image - anchor)
            quality = metadata.get("final_check", metadata.get("best", {}))
            stop = ready.get("early_stop", metadata.get("early_stop", {}))
            actual_steps = metadata.get("steps_completed", max(row["step"] for row in metadata["history"]))
            if name in PROFILES and (stop.get("rule_satisfied") is not True
                                     or stop.get("steps_completed") != actual_steps
                                     or metadata.get("early_stop") != stop):
                raise ValueError("外观报告未绑定真实自适应停止证据")
            record = dict(steps_completed=actual_steps, steps_maximum=1000,
                stop_reason=stop.get("reason", "fixed_schedule_max_steps"),
                early_stop=stop, source_ssim=quality["source_ssim"],
                foreground_ssim=quality.get("foreground_ssim"), background_ssim=quality.get("background_ssim"),
                source_mae_levels=float(difference.mean()),
                foreground_mae_levels=float(difference[foreground].mean()),
                background_mae_levels=float(difference[~foreground].mean()),
                foreground_added_mae_levels=float(addition[foreground].mean()),
                background_added_mae_levels=float(addition[~foreground].mean()),
                foreground_max_levels=int(addition[foreground].max()),
                background_max_levels=int(addition[~foreground].max()),
                foreground_fraction=float(foreground.mean()), halo_error=quality.get("halo_error"),
                best_step=metadata["best"]["step"], accepted_updates=metadata["accepted_updates"],
                rejected_updates=metadata["rejected_updates"], image=asset["image"], image_sha256=asset["sha256"],
                metadata=str(metadata_path), metadata_sha256=sha256_file(metadata_path))
            record["worst_patch32"] = {
                reference: {region_name: _maximum_patch_mae(error, mask)
                            for region_name, mask in (("whole", None), ("foreground", foreground), ("background", ~foreground))}
                for reference, error in (("source_original", difference), ("harmonized_anchor", addition))}
            if name in PROFILES and any(record[field] is None for field in ("foreground_ssim", "background_ssim")):
                raise ValueError("分布式生成缺少前景或背景结构相似度记录")
            for field in ("source_ssim", "source_mae_levels", "foreground_mae_levels", "background_mae_levels"):
                if not math.isfinite(record[field]):
                    raise ValueError("外观统计包含非有限数值")
            records[name][key] = record
            for path in (metadata_path, asset["image"], asset["ready"], source_path, anchor_path, foreground_path):
                bindings[str(path)] = sha256_file(path)
    freeze_json(run / "GENERATION_QUALITY.json", dict(records=records, files=bindings,
        references=dict(mae="source_original", added_mae_and_max="frozen_harmonized_anchor"),
        source_code_sha256=sha256_file(__file__)))
    return records


def _plot_comparison(run, quality):
    """科学图像对照仅按原宽高比展示，不改动原始图片内容。"""
    plan = read_json(run / "EXPERIMENT_PLAN.json")
    registry = read_json(run / "FROZEN_IMAGES.json")
    previous = read_json(RANKFOCUS_RUN / "FROZEN_IMAGES.json")
    key = GROUPS[0]
    paths = [plan["generation_inputs"][key]["definition"]["data"]["source"],
             previous["images"]["rankfocus"][key]["image"],
             *[registry["images"][name][key]["image"] for name in PROFILES]]
    signature = dict(images={str(path): sha256_file(path) for path in paths},
                     quality_sha256=sha256_file(run / "GENERATION_QUALITY.json"),
                     plan_sha256=sha256_file(run / "EXPERIMENT_PLAN.json"))
    root = run / "analysis"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "image_comparison.png"
    certificate = root / "IMAGE_COMPARISON.json"
    if certificate.exists():
        old = read_json(certificate)
        if old.get("inputs") != signature:
            raise ValueError("既有图片对照与冻结图片或质量统计不一致")
        _check(path, old["image_sha256"])
        return dict(path=str(path), sha256=old["image_sha256"])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from PIL import Image
    font = Path("C:/Windows/Fonts/msyh.ttc")
    if font.is_file():
        font_manager.fontManager.addfont(str(font))
        family = font_manager.FontProperties(fname=str(font)).get_name()
    else:
        family = "sans-serif"
    labels = ["源商品原图", "上一轮高收益强攻档", "仅前景同预算对照", "前景与背景均衡档", "前景与背景宽松档"]
    with plt.rc_context({"font.family": family, "axes.unicode_minus": False}):
        fig, axes = plt.subplots(1, 5, figsize=(18, 6), facecolor="white")
        for index, (axis, image_path, title) in enumerate(zip(axes, paths, labels)):
            axis.imshow(Image.open(image_path).convert("RGB"))
            axis.axis("off")
            if index:
                name = ("rankfocus", *PROFILES)[index - 1]
                row = quality[name][key]
                title += (f"\n实际{row['steps_completed']}步"
                          f"\n前景平均变化 {row['foreground_mae_levels']:.2f} 级"
                          f"\n背景平均变化 {row['background_mae_levels']:.2f} 级")
            axis.set_title(title, fontsize=11, pad=10)
        fig.suptitle("第七组冻结图片对照：观察主体纹理、背景及交界处的可见变化", fontsize=18, y=.97)
        fig.text(.5, .025,
            "所有图片仅按原宽高比展示；平均变化均相对同组源原图。源商品为洗手液，目标品类为洗发水。",
            ha="center", fontsize=11, color="#444444")
        fig.subplots_adjust(left=.015, right=.995, top=.78, bottom=.10, wspace=.08)
        fig.savefig(str(path), dpi=170, facecolor="white")
        plt.close(fig)
    freeze_json(certificate, dict(inputs=signature, image_sha256=sha256_file(path)))
    return dict(path=str(path), sha256=sha256_file(path))


def report(run):
    """报告所有预注册档位及真实停止步数，不按正式排名选图。"""
    run = Path(run).resolve()
    inputs = _inputs(run)
    verification = _validate_verification(run, inputs)
    results = {name: {key: read_json(_result_path(run, name, key)) for key in GROUPS} for name in BANKS}
    quality = _quality_records(run)
    comparison_image = _plot_comparison(run, quality)
    summary = dict(schema_version=1, status="completed", scope="single_group_retrieval_pilot",
        groups=list(GROUPS), query_mode="joint", pool_count=189, simultaneous_replacements=1,
        query_count_per_bank=20, shopping_executed=False, automatic_other_groups_enabled=False,
        inputs_sha256=sha256_file(run / "EVALUATION_INPUTS.json"),
        verification_sha256=sha256_file(run / "INDEPENDENT_VERIFICATION.json"),
        generation_quality_sha256=sha256_file(run / "GENERATION_QUALITY.json"), generation_quality=quality,
        comparison_image=comparison_image, banks={})
    references = ("clean", "baseline", "lucid", "rankfocus", "foreground_control")
    for name in BANKS:
        all_rows = [dict(row, task_id=key + ":" + row["task_id"])
                    for key in GROUPS for row in results[name][key]["rankings"]]
        sections = {key: results[name][key]["metrics"] for key in GROUPS}
        sections["overall"] = _metrics([row["source_rank"] for row in all_rows])
        comparisons = {}
        for reference in references:
            comparisons[reference] = {}
            for key in (*GROUPS, "overall"):
                previous = ([dict(row, task_id=g + ":" + row["task_id"])
                             for g in GROUPS for row in results[reference][g]["rankings"]]
                            if key == "overall" else results[reference][key]["rankings"])
                current = all_rows if key == "overall" else results[name][key]["rankings"]
                before, after = _metrics([row["source_rank"] for row in previous]), sections[key]
                comparisons[reference][key] = dict(**_paired(current, previous),
                    mean_rank_reduction=before["mean_rank"] - after["mean_rank"],
                    mrr_gain=after["mrr"] - before["mrr"],
                    **{f"top{k}_gain": after[f"top{k}"] - before[f"top{k}"] for k in (1, 3, 5)},
                    **{f"top{k}_rate_gain": after[f"top{k}_rate"] - before[f"top{k}_rate"] for k in (1, 3, 5)})
        summary["banks"][name] = dict(metrics=sections, comparisons=comparisons)
    for name in PROFILES:
        for key in GROUPS:
            control, current, historical = quality["foreground_control"][key], quality[name][key], quality["rankfocus"][key]
            summary["banks"][name].setdefault("quality_comparisons", {})[key] = dict(
                foreground_mae_reduction_vs_control=control["foreground_mae_levels"] - current["foreground_mae_levels"],
                background_mae_increase_vs_control=current["background_mae_levels"] - control["background_mae_levels"],
                foreground_mae_reduction_vs_rankfocus=historical["foreground_mae_levels"] - current["foreground_mae_levels"],
                background_mae_increase_vs_rankfocus=current["background_mae_levels"] - historical["background_mae_levels"],
                actual_steps=current["steps_completed"], steps_saved_vs_1000=1000-current["steps_completed"],
                stop_reason=current["stop_reason"])
    freeze_json(run / "SUMMARY.json", summary)
    labels = {"clean": "原图库", "baseline": "原无像素感知方法", "lucid": "原始像素方法",
              "rankfocus": "上一轮高收益强攻档", "foreground_control": "仅前景同预算对照",
              "distributed_balanced": "前景与背景均衡档", "distributed_flexible": "前景与背景宽松档"}
    lines = ["# 第七组分布式潜变量编辑先导实验", "",
        "本轮仅运行第七组洗发水目标组，不运行第六组，也不自动追加其他组。源商品为洗手液，目标品类为洗发水。",
        "三个新档位全部完整执行预注册的自适应日程并冻结后，才打开共享池及正式查询。",
        "每个修改库只替换本组同一条图片，其余188条与全部文字保持不变；共七个完整189条全局库。",
        "每库20条原特点文字加同一张原独立模糊照片，查询和记忆均图文各半，仅无防御；没有执行购物。", "",
        "|图库|请求数|第一|前三|前五|平均名次↓|平均倒数排名↑|",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for name in BANKS:
        m = summary["banks"][name]["metrics"]["overall"]
        cells = [f"{m[f'top{k}']}/{m['query_count']}（{m[f'top{k}_rate']:.1%}）" for k in (1,3,5)]
        lines.append(f"|{labels[name]}|{m['query_count']}|{'|'.join(cells)}|{m['mean_rank']:.3f}|{m['mrr']:.5f}|")
    lines += ["", "|新档位|对照|前三增量|配对新增前三|配对丢失前三|平均名次改善|",
              "|---|---|---:|---:|---:|---:|"]
    for name in PROFILES:
        for reference in references:
            delta = summary["banks"][name]["comparisons"][reference]["overall"]
            lines.append(f"|{labels[name]}|{labels[reference]}|{delta['top3_gain']:+d}|{delta['top3_gained']}|{delta['top3_lost']}|{delta['mean_rank_reduction']:+.3f}|")
    lines += ["", "|档位|实际优化步数|停止原因|原图整体结构相似度|前景结构相似度|背景结构相似度|前景平均变化，级|背景平均变化，级|",
              "|---|---:|---|---:|---:|---:|---:|---:|"]
    reason_labels = {"plateau": "公开训练目标进入平台", "max_steps": "达到最大步数",
                     "fixed_schedule_max_steps": "历史固定千步"}
    for name in ("rankfocus", *PROFILES):
        row = quality[name][GROUPS[0]]
        foreground_ssim = "未记录" if row["foreground_ssim"] is None else f"{row['foreground_ssim']:.5f}"
        background_ssim = "未记录" if row["background_ssim"] is None else f"{row['background_ssim']:.5f}"
        lines.append(f"|{labels[name]}|{row['steps_completed']}|{reason_labels.get(row['stop_reason'],row['stop_reason'])}|{row['source_ssim']:.5f}|{foreground_ssim}|{background_ssim}|{row['foreground_mae_levels']:.3f}|{row['background_mae_levels']:.3f}|")
    lines += ["", "前景和背景平均变化均相对同组源原图逐区域归一化。另在外观数据记录中保存相对冻结融合起点的新增变化及区域最大值，避免混淆参照。",
        "", "![源原图、旧强攻与三档新图对照](analysis/image_comparison.png)", "",
        "|档位|原图参照最差32×32块平均变化|原图参照最差前景块|原图参照最差背景块|融合起点参照最差32×32块|",
        "|---|---:|---:|---:|---:|"]
    def patch_cell(value):
        return "无足够覆盖区域" if value is None else f"{value['mae_levels']:.3f}级，左上角{value['position_xy']}"
    for name in ("rankfocus", *PROFILES):
        patches = quality[name][GROUPS[0]]["worst_patch32"]
        source, anchor = patches["source_original"], patches["harmonized_anchor"]
        lines.append(f"|{labels[name]}|{patch_cell(source['whole'])}|{patch_cell(source['foreground'])}|{patch_cell(source['background'])}|{patch_cell(anchor['whole'])}|")
    lines += ["", "最差局部块以32×32窗口、一步一像素滑动统计；前景或背景覆盖至少80%的窗口才计入对应区域，并按区域内像素归一化。完整位置、区域覆盖率和两种参照的全部统计保存在外观数据记录中。",
        "前景同预算对照与均衡档只改变背景能否参与编辑；宽松档还放宽预算，须分别解释区域范围与预算因素。",
        "停止只由公开训练目标的固定窗口规则或最大步数决定；完整完成自适应日程不等于全部执行1000步。真实步数、每窗口增益和停止原因保存在各生成目录的停止凭证中。", "",
        f"独立新进程、新编码器核对140个完整排名及26460个候选分数，最大误差{verification['max_score_error']:.9g}，容差为百万分之一。", "",
        "本轮是单组先导实验，且该组先前结果已用于开发，不能外推到其他组或声称独立盲测。",
        "评估编码器与训练代理重合，不声称独立留出黑盒迁移。正式20条特点文字中有一条与训练句精确重合。",
        "前三、前五是源记忆检索比例，不是购买或误购率。共享池保持真实购买与未购买参考标识，不伪造购买。",
        "背景也可优化只意味着分散变化的可能性，不能预先保证更自然；须结合分区域指标及实际图片观察判断。",
        "旧方法、旧图片、旧结果及既有购物状态保持只读。", ""]
    (run / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


