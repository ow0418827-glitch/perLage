"""两组独立无像素放宽实验：四图冻结后统一检索，独立进程复算全部分数。"""
from __future__ import annotations

import copy
import math
import os

from src.paths import PROJECT, EXPERIMENT_BASE_GROUPS9 as BASE, ProjectPath as Path
from src.visual.audit import digest, freeze_json, read_json
from src.visual.image_utils import sha256_file


GROUPS = ("06_jeans", "07_shampoo")
PROFILES = ("relaxed", "strong")
BANKS = ("clean", "baseline", *PROFILES)
PREVIOUS = PROJECT / "output/perlage_no_pixel_groups9_20261109"
QUERY_RUN = PROJECT / "output/groups9_multimodal_20261110"
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
        raise ValueError("实验必须预先冻结第六、七组及两个放宽档位")
    registry_file = run / "FROZEN_IMAGES.json"
    frozen = read_json(registry_file)
    if frozen.get("status") != "frozen" or set(frozen.get("images", {})) != set(PROFILES):
        raise ValueError("两个档位四张完整图片必须全部冻结后评估")
    if frozen.get("plan_sha256") != sha256_file(plan_file):
        raise ValueError("冻结图片未绑定预定训练计划")
    for profile in PROFILES:
        if set(frozen["images"][profile]) != set(GROUPS):
            raise ValueError("冻结图的两组范围不完整")
        for key in GROUPS:
            item = frozen["images"][profile][key]
            _check(item["image"], item["sha256"])
            _check(item["ready"], item["ready_sha256"])
            ready = read_json(item["ready"])
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
            if (type(latent.get("steps_completed")) is not int or latent["steps_completed"] < 1
                    or latent["steps_completed"] != latent.get("steps_requested")):
                raise ValueError("最终潜变量细化没有完成预定步数")
            final = ready.get("candidates", {}).get("final", {})
            if final.get("sha256") != item["sha256"]:
                raise ValueError("冻结图与完整方法最终候选不是同一图片")
    return frozen


def build_banks(entries, pool_path, replacements):
    """保留完整共享池；每个修改库同时只替换相同两条图片。"""
    from src.visual.visual_retrieval import resolve_image
    if len(entries) != 189 or len({e["Id"] for e in entries}) != 189:
        raise ValueError("必须使用完整且唯一的一百八十九条共享池")
    if set(replacements) != set(BANKS) - {"clean"}:
        raise ValueError("替换清单必须包括原无像素基线和两个放宽档位")
    ids = set(replacements["baseline"])
    if len(ids) != 2 or any(set(row) != ids for row in replacements.values()):
        raise ValueError("三个修改库必须同时替换相同的两条源记忆")
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
        if stripped != clean or sum("RetrievalImage" in entry for entry in bank) != 2:
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
        for profile in PROFILES:
            asset = frozen["images"][profile][key]
            ready = read_json(asset["ready"])
            if ready.get("source_sha256") != meta["source_sha256"] or ready.get("target_sha256") != meta["target_sha256"]:
                raise ValueError("放宽实验完整图与本轮冻结图对不一致")
            replacements[profile][source_id] = dict(image=asset["image"], sha256=asset["sha256"])
            bindings[asset["image"]] = asset["sha256"]
            bindings[asset["ready"]] = asset["ready_sha256"]
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
                  simultaneous_replacements_per_modified_bank=2, expected_rankings=160, expected_score_comparisons=30240,
                  baseline_policy="recompute_original_perlage_images_for_both_groups_in_same_complete_pool",
                  historical_rankings_used_for_selection=False, input_read_only_after_all_four_complete_images=True)
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
    if (certificate.get("status") != "completed" or certificate.get("rankings") != 160
            or certificate.get("score_count") != 30240 or certificate.get("execution_device") != "cuda"
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
    if (verification.get("status") != "passed" or verification.get("score_comparisons") != 30240
            or verification.get("rankings_recomputed") != 160 or verification.get("fresh_process") is not True
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
    """四图全冻结后，新编码四个全局库和同样四十条图文查询。"""
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
                inputs_sha256=sha256_file(run / "EVALUATION_INPUTS.json"), rankings=160, score_count=30240,
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
    if compared != 30240 or rankings != 160:
        raise ValueError("独立复算没有覆盖全部一百六十排名及三万零二百四十分数")
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


def report(run):
    """报告所有预定档位，不按正式名次挑选最终图片。"""
    run = Path(run).resolve()
    inputs = _inputs(run)
    verification = _validate_verification(run, inputs)
    results = {name: {key: read_json(_result_path(run, name, key)) for key in GROUPS} for name in BANKS}
    summary = dict(schema_version=1, status="completed", scope="retrieval_only", query_mode="joint", pool_count=189,
                   simultaneous_replacements=2, query_count_per_bank=40, shopping_executed=False,
                   inputs_sha256=sha256_file(run / "EVALUATION_INPUTS.json"),
                   verification_sha256=sha256_file(run / "INDEPENDENT_VERIFICATION.json"), banks={})
    for name in BANKS:
        all_rows = [dict(row, task_id=key + ":" + row["task_id"]) for key in GROUPS for row in results[name][key]["rankings"]]
        sections = {key: results[name][key]["metrics"] for key in GROUPS}
        sections["overall"] = _metrics([r["source_rank"] for r in all_rows])
        comparisons = {}
        for reference in ("clean", "baseline"):
            comparisons[reference] = {}
            for key in (*GROUPS, "overall"):
                previous = ([dict(row, task_id=g + ":" + row["task_id"]) for g in GROUPS
                             for row in results[reference][g]["rankings"]] if key == "overall"
                            else results[reference][key]["rankings"])
                current = all_rows if key == "overall" else results[name][key]["rankings"]
                before, after = _metrics([r["source_rank"] for r in previous]), sections[key]
                comparisons[reference][key] = dict(**_paired(current, previous),
                    mean_rank_reduction=before["mean_rank"] - after["mean_rank"], mrr_gain=after["mrr"] - before["mrr"],
                    **{f"top{k}_gain": after[f"top{k}"] - before[f"top{k}"] for k in (1, 3, 5)},
                    **{f"top{k}_rate_gain": after[f"top{k}_rate"] - before[f"top{k}_rate"] for k in (1, 3, 5)})
        summary["banks"][name] = dict(metrics=sections, comparisons=comparisons)
    freeze_json(run / "SUMMARY.json", summary)
    labels = {"clean": "原图库", "baseline": "原无像素感知方法", "relaxed": "放宽档", "strong": "强放宽档"}
    lines = ["# 第六、七组无像素感知方法放宽实验", "",
             "训练保持原有特点文字条件。四张完整图片全部冻结后，才读取完整189条共享池及正式查询。",
             "每个修改库同时替换牛仔裤和洗发水两条图片，其余187条与原图库相同；原无像素基线也按此口径重新计算。",
             "检索只用无防御条件，每组20条既有特点文字加同一张独立模糊照片，查询和记忆均图文各半；没有执行购物。", "",
             "|图库|范围|请求数|第一|前三|前五|平均名次↓|平均倒数排名↑|",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for name in BANKS:
        for key in (*GROUPS, "overall"):
            metrics = summary["banks"][name]["metrics"][key]
            cells = [f"{metrics[f'top{k}']}/{metrics['query_count']}（{metrics[f'top{k}_rate']:.1%}）" for k in (1, 3, 5)]
            scope = "两组合计" if key == "overall" else inputs["groups"][key]["name"]
            lines.append(f"|{labels[name]}|{scope}|{metrics['query_count']}|{'|'.join(cells)}|{metrics['mean_rank']:.3f}|{metrics['mrr']:.5f}|")
    lines += ["", "|档位|对照|范围|前三增加数|配对新增前三|配对丢失前三|平均名次改善|倒数排名增量|", "|---|---|---|---:|---:|---:|---:|---:|"]
    for name in PROFILES:
        for reference in ("clean", "baseline"):
            for key in (*GROUPS, "overall"):
                delta = summary["banks"][name]["comparisons"][reference][key]
                scope = "两组合计" if key == "overall" else inputs["groups"][key]["name"]
                lines.append(f"|{labels[name]}|{labels[reference]}|{scope}|{delta['top3_gain']:+d}|{delta['top3_gained']}|{delta['top3_lost']}|{delta['mean_rank_reduction']:+.3f}|{delta['mrr_gain']:+.5f}|")
    lines += ["", "独立进程重新编码所有图文输入，已核对160个完整排名和30240个候选分数；最大误差为"
              f"{verification['max_score_error']:.9g}，容差为百万分之一。", "",
              "本轮是预定两组、预定两个档位的探索实验，报告全部结果，不根据正式排名换图、换种子或选优。",
              "评估编码器按基础包显式使用与训练代理重合的公开模型，不声称独立留出黑盒迁移。",
              "每组二十条正式文字中有一条与训练正文精确重合，不能把查询文字视为完全独立的语义留出集。",
              "第一、前三、前五为源记忆检索比例，不是购买或误购率。原商品参考仍标识为未观察到购买。",
              "已有九组购物暂停标记及检查点没有在本轮更改。", ""]
    report_path = run / "REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return summary
