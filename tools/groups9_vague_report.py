"""冻结九组模糊图文检索报告；只读取已完成排名与独立复核证据。"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
from statistics import mean

from src.visual.audit import digest, freeze_json, read_json


PROJECT = Path(__file__).resolve().parents[1]
OLD_JOINT = PROJECT / "output/groups9_multimodal_20261110"
OLD_TEXT = {
    "lucid": (PROJECT / "output/lucid_groups9_20261109", "lucid"),
    "perlage_no_pixel": (PROJECT / "output/perlage_no_pixel_groups9_20261109", "PerLAGE_no_pixel"),
}
METHOD_NAMES = {"lucid": "原始像素方法", "perlage_no_pixel": "无像素感知方法"}
VARIANT_NAMES = {"vague_joint": "本次模糊图文", "feature_joint": "最近特点图文", "feature_text": "最近特点纯文字"}
BANK_NAMES = {"clean": "原图库", "modified": "修改库"}


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _stats(ranks: list[int]) -> dict:
    if not ranks or any(type(x) is not int or not 1 <= x <= 189 for x in ranks):
        raise ValueError("报告名次为空、类型无效或超出一百八十九条库")
    count = len(ranks)
    return dict(query_count=count, top1=sum(x == 1 for x in ranks),
                top3=sum(x <= 3 for x in ranks), top5=sum(x <= 5 for x in ranks),
                top1_rate=sum(x == 1 for x in ranks) / count,
                top3_rate=sum(x <= 3 for x in ranks) / count,
                top5_rate=sum(x <= 5 for x in ranks) / count,
                mean_rank=mean(ranks), source_ranks=ranks)


def _difference(after: dict, before: dict) -> dict:
    if after["query_count"] != before["query_count"]:
        raise ValueError("比较两边请求数量不一致")
    return dict(query_count=after["query_count"],
                top1_count_delta=after["top1"] - before["top1"],
                top3_count_delta=after["top3"] - before["top3"],
                top5_count_delta=after["top5"] - before["top5"],
                top1_percentage_point_delta=100 * (after["top1_rate"] - before["top1_rate"]),
                top3_percentage_point_delta=100 * (after["top3_rate"] - before["top3_rate"]),
                top5_percentage_point_delta=100 * (after["top5_rate"] - before["top5_rate"]),
                mean_rank_delta=after["mean_rank"] - before["mean_rank"])


def _paired(clean: list[int], modified: list[int]) -> dict:
    if len(clean) != len(modified):
        raise ValueError("同查询原图修改图名次数量不一致")
    return {f"top{k}": dict(
        gained=sum(a > k and b <= k for a, b in zip(clean, modified)),
        lost=sum(a <= k and b > k for a, b in zip(clean, modified)),
        retained=sum(a <= k and b <= k for a, b in zip(clean, modified)),
        neither=sum(a > k and b > k for a, b in zip(clean, modified))) for k in (1, 3, 5)}


def _rank_rows(result: dict, source_id: str, queries: list[dict], bank_sha: str) -> list[int]:
    rows = result.get("rankings", [])
    if len(rows) != len(queries) or len(rows) != 20:
        raise ValueError("报告须包含二十个完整请求")
    protocol = result["protocol"]
    if (protocol.get("bank_sha256") != bank_sha or protocol.get("key_mode") != "joint"
            or protocol.get("query_mode") != "joint" or protocol.get("index_text") != "product_feature"
            or protocol.get("key_image_weight") != 0.5 or protocol.get("query_image_weight") != 0.5):
        raise ValueError("报告的库、检索键或图文权重发生变化")
    ranks = []
    for query, row in zip(queries, rows):
        if row.get("task_id") != query["task_id"]:
            raise ValueError("报告逐请求身份不一致")
        ranking = row.get("ranking", [])
        if len(ranking) != 189 or len({x["Id"] for x in ranking}) != 189:
            raise ValueError("报告未包含完整一百八十九条不同记忆")
        if [x["rank"] for x in ranking] != list(range(1, 190)):
            raise ValueError("报告完整名次不是一至一百八十九")
        expected_text = hashlib.sha256(query["instruction"].encode("utf-8")).hexdigest()
        for candidate in ranking:
            if (candidate.get("query_text_sha256") != expected_text
                    or candidate.get("query_image_sha256") != query["query_image_sha256"]
                    or candidate.get("bank_sha256") != bank_sha
                    or candidate.get("model_sha256") != protocol["model_sha256"]):
                raise ValueError("候选分数未绑定本次查询、图库或实际模型")
            if not math.isfinite(candidate["score"]):
                raise ValueError("候选分数出现非有限数")
        actual = next(x for x in ranking if x["Id"] == source_id)
        if actual["rank"] != row["source_rank"]:
            raise ValueError("源商品名次与完整排名不符")
        if abs(actual["score"] - row["source_score"]) > 1e-7:
            raise ValueError("源商品分数与完整排名不符")
        ranks.append(actual["rank"])
    return ranks


def _check_joint_controls(current: dict, previous: dict, image_sha: str, bank_sha: str) -> None:
    for field in ("model_sha256", "image_decoder", "key_mode", "query_mode", "index_text",
                  "index_text_policy", "key_image_weight", "query_image_weight"):
        if current["protocol"].get(field) != previous["protocol"].get(field):
            raise ValueError("与最近图文比较时检索设置不一致：" + field)
    if previous["protocol"].get("bank_sha256") != bank_sha:
        raise ValueError("与最近图文比较时完整图库不一致")
    if len(previous.get("rankings", [])) != 20:
        raise ValueError("最近图文结果不是二十条请求")
    current_candidates = {r["Id"]: r for r in current["rankings"][0]["ranking"]}
    for row in previous["rankings"]:
        if len(row["ranking"]) != 189 or {r["Id"] for r in row["ranking"]} != set(current_candidates):
            raise ValueError("与最近图文比较时候选商品集合不一致")
        for previous_candidate in row["ranking"]:
            present = current_candidates[previous_candidate["Id"]]
            if any(previous_candidate.get(k) != present.get(k)
                   for k in ("image_sha256", "text_sha256", "model_sha256", "bank_sha256")):
                raise ValueError("与最近图文比较时记忆图片、文字或模型改变")
            if previous_candidate.get("query_image_sha256") != image_sha:
                raise ValueError("与最近图文比较时查询照片改变")


def _freeze_text(path: Path, content: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise ValueError("已有冻结报告内容不同，拒绝覆盖：" + str(path))
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _cell(stats: dict, k: int) -> str:
    return f"{stats[f'top{k}']}/{stats['query_count']}（{stats[f'top{k}_rate']:.2%}）"


def write_report(run: Path, plan: dict) -> dict:
    """校验两方法完成证据，冻结中文报告与完成标记，返回机器可读汇总。"""
    run = Path(run).resolve()
    if not run.is_relative_to(PROJECT / "output"):
        raise ValueError("报告只能写入当前项目独立输出目录")
    plan_file = run / "SCHEDULED_INPUTS.json"
    plan_sha = _sha(plan_file)
    if read_json(plan_file) != plan or plan.get("shopping_enabled") is not False or plan.get("generation_enabled") is not False:
        raise ValueError("报告输入不是本轮冻结的只检索计划")
    if set(plan.get("methods", {})) != set(METHOD_NAMES):
        raise ValueError("报告必须包含指定的两种方法")
    inputs = {str(plan_file): plan_sha}

    def load(path: Path) -> object:
        path = Path(path).resolve()
        inputs[str(path)] = _sha(path)
        return read_json(path)

    image_hashes = {}

    def key_evidence(bank_path: Path, bank: list[dict]) -> dict:
        from src.visual.visual_retrieval import memory_index_text, resolve_image
        evidence = {}
        for entry in bank:
            image_path = Path(resolve_image(entry, bank_path)).resolve()
            if str(image_path) not in image_hashes:
                image_hashes[str(image_path)] = _sha(image_path)
                inputs[str(image_path)] = image_hashes[str(image_path)]
            evidence[entry["Id"]] = dict(
                image_sha256=image_hashes[str(image_path)],
                text_sha256=hashlib.sha256(memory_index_text(entry, "product_feature").encode("utf-8")).hexdigest())
        if len(evidence) != 189:
            raise ValueError("比较文字条件的完整记忆库不是一百八十九条不同记忆")
        return evidence

    for binding in (plan["old_text_origin"], plan["frozen_current_plan"]):
        load(Path(binding["path"]))
        if inputs[str(Path(binding["path"]).resolve())] != binding["sha256"]:
            raise ValueError("绑定来源计划摘要改变")
    load(Path(plan["encoder_config"]))
    if inputs[str(Path(plan["encoder_config"]).resolve())] != plan["encoder_config_sha256"]:
        raise ValueError("绑定编码器配置改变")
    current_plan = read_json(plan["frozen_current_plan"]["path"])
    methods, certificates = {}, {}
    ranking_count = score_count = 0
    for method in METHOD_NAMES:
        groups = plan["methods"][method]["groups"]
        if len(groups) != 9 or len({g["key"] for g in groups}) != 9:
            raise ValueError("每方法必须包含九个不同商品组")
        cert_file = run / "methods" / method / "INDEPENDENT_VERIFICATION.json"
        cert = load(cert_file)
        if (cert.get("status") != "passed" or cert.get("score_comparisons") != 68040
                or cert.get("rankings_recomputed") != 360 or cert.get("plan_sha256") != plan_sha):
            raise ValueError("方法独立复核证书不完整或不属于本轮计划")
        if not math.isfinite(cert.get("max_score_error", math.inf)) or cert["max_score_error"] > 1e-6:
            raise ValueError("独立复核分数误差超过容差")
        bindings = {str(Path(p).resolve()): h for p, h in cert.get("files", {}).items()}
        for filename, expected in bindings.items():
            if _sha(Path(filename)) != expected:
                raise ValueError("独立复核绑定结果已改变：" + filename)
        certificates[method] = dict(path=str(cert_file), sha256=inputs[str(cert_file)],
            score_comparisons=cert["score_comparisons"], rankings_recomputed=cert["rankings_recomputed"],
            max_score_error=cert["max_score_error"])
        score_count += cert["score_comparisons"]
        group_rows = []
        aggregate = {variant: {bank: [] for bank in BANK_NAMES} for variant in VARIANT_NAMES}
        for group in groups:
            metrics = {variant: {} for variant in VARIANT_NAMES}
            previous_group = next(g for g in current_plan["methods"][method]["groups"] if g["key"] == group["key"])
            if (previous_group["source_memory_id"] != group["source_memory_id"]
                    or previous_group["frozen_image_sha256"] != group["frozen_image_sha256"]):
                raise ValueError("与最近图文比较时源商品或冻结修改图改变")
            text_root, text_method = OLD_TEXT[method]
            text_dir = text_root / "groups" / group["key"] / "methods" / text_method / "evaluation"
            text_plan = load(text_dir / "evaluation_plan.json")
            text_results = load(text_dir / "evaluation.json")
            if (text_plan.get("source_memory_id") != group["source_memory_id"]
                    or text_plan.get("frozen_image_sha256") != group["frozen_image_sha256"]
                    or text_plan.get("pool_count") != 189):
                raise ValueError("最近纯文字比较的商品、图片或库规模不一致")
            for bank in BANK_NAMES:
                path = run / "methods" / method / "retrieval" / group["key"] / (bank + ".json")
                current = load(path)
                if bindings.get(str(path)) != inputs[str(path)]:
                    raise ValueError("当前结果未被独立证书逐项绑定")
                ranks = _rank_rows(current, group["source_memory_id"], group["queries"], group["banks"][bank]["sha256"])
                ranking_count += len(ranks)
                previous = load(OLD_JOINT / "methods" / method / "retrieval" / group["key"] / (bank + ".json"))
                _check_joint_controls(current, previous, group["queries"][0]["query_image_sha256"], group["banks"][bank]["sha256"])
                previous_ranks = [q["source_rank"] for q in previous["rankings"]]
                text_result = next(r for r in text_results if r["bank"] == bank and r["condition"]["kind"] == "none")
                if text_result["retrieval_protocol"].get("model_sha256") != current["protocol"]["model_sha256"]:
                    raise ValueError("最近纯文字结果使用了不同模型")
                # 购物适配副本与原评估库的非检索字段或路径表示可以不同；
                # 不能以整个文件字节不同误判商品图片和检索正文不同。
                present_bank_path = Path(group["banks"][bank]["path"])
                present_bank = load(present_bank_path)
                text_bank_path = Path(text_result["bank_path"])
                text_bank = load(text_bank_path)
                if inputs[str(text_bank_path.resolve())] != text_result["bank_sha256"]:
                    raise ValueError("最近纯文字结果绑定的原评估库文件改变")
                if key_evidence(present_bank_path, present_bank) != key_evidence(text_bank_path, text_bank):
                    raise ValueError("最近纯文字比较的完整记忆图片或文字键不一致")
                text_ranks = [q["rank"] for q in text_result["queries"]]
                if len(text_ranks) != 20 or len(previous_ranks) != 20:
                    raise ValueError("历史比较条件请求数不符")
                for variant, values in (("vague_joint", ranks), ("feature_joint", previous_ranks), ("feature_text", text_ranks)):
                    metrics[variant][bank] = _stats(values)
                    aggregate[variant][bank].extend(values)
            group_rows.append(dict(group=group["key"], name=group["name"], source_memory_id=group["source_memory_id"], metrics=metrics,
                modified_minus_clean={v: _difference(metrics[v]["modified"], metrics[v]["clean"]) for v in VARIANT_NAMES},
                current_clean_modified_paired=_paired(metrics["vague_joint"]["clean"]["source_ranks"], metrics["vague_joint"]["modified"]["source_ranks"]),
                current_minus_feature_joint={b: _difference(metrics["vague_joint"][b], metrics["feature_joint"][b]) for b in BANK_NAMES},
                current_minus_feature_text={b: _difference(metrics["vague_joint"][b], metrics["feature_text"][b]) for b in BANK_NAMES}))
        overall = {v: {b: _stats(aggregate[v][b]) for b in BANK_NAMES} for v in VARIANT_NAMES}
        methods[method] = dict(groups=group_rows, overall=overall,
            modified_minus_clean={v: _difference(overall[v]["modified"], overall[v]["clean"]) for v in VARIANT_NAMES},
            current_clean_modified_paired=_paired(aggregate["vague_joint"]["clean"], aggregate["vague_joint"]["modified"]),
            current_minus_feature_joint={b: _difference(overall["vague_joint"][b], overall["feature_joint"][b]) for b in BANK_NAMES},
            current_minus_feature_text={b: _difference(overall["vague_joint"][b], overall["feature_text"][b]) for b in BANK_NAMES})
    if ranking_count != 720 or score_count != 136080:
        raise ValueError("两方法完整排名或独立候选分数复核数量不足")
    for filename, expected in inputs.items():
        if _sha(Path(filename)) != expected:
            raise ValueError("报告期间读取的证据发生变化：" + filename)
    summary = dict(schema_version=1, status="completed", scope="retrieval_only_no_defense", plan_sha256=plan_sha,
        rankings=ranking_count, score_comparisons=score_count, pool_count=189, methods=methods, independent_verification=certificates,
        controls_verified=["same_frozen_method_images", "same_189_entry_bank_bytes_as_previous_joint", "same_model_and_image_decoder", "same_memory_image_and_text_hashes_in_joint_and_text_comparisons", "same_query_photo_hashes_as_previous_joint"],
        comparison_scope="same_20_request_counts_aggregate_comparison_not_identical_utterance_pairs",
        limitations=["本次只统计指定源记忆的检索名次，不是最终购物或误购率。", "本次逐字复用旧版模糊文字，沿用当前九张冻结照片，并非旧照片和旧模型的完整复现。", "历史特点文字与本次模糊文字不同，只比较每组二十请求的聚合统计，不声称逐句语义相同。", "本轮旧版模糊文字为中文，最近特点文字为英文；跨条件同时改变语言、措辞内容和具体程度，不能把差异单独归因于模糊程度或单独语言效应。实际请求按用户指定旧句逐字保留。", "每组二十种措辞共用一张照片，不能作为二十个独立视觉情境。", "指定评估编码器与训练代理重合，不声称独立留出黑盒迁移；第四、八组保留历史选图门槛未通过状态。"],
        input_files=inputs)
    lines = ["# 九组旧版模糊图文检索复测", "", "本轮仅无防御检索，原始像素方法与无像素感知方法串行执行。复用旧版二十条通用中文模糊文字、当前九张冻结独立无品牌照片及既有完整成图；每次检索完整一百八十九条共享记忆。", "", "两方法独立复核全部通过：七百二十条全库排名、十三万六千零八十个候选分数。以下统计指定源记忆的名次，不能解释为最终误购率。", "", "## 总体结果与历史条件", "", "|方法|查询条件|图库|第一|前三|前五|平均名次|", "|---|---|---|---:|---:|---:|---:|"]
    for method, result in methods.items():
        for variant in VARIANT_NAMES:
            for bank in BANK_NAMES:
                s = result["overall"][variant][bank]
                lines.append(f"|{METHOD_NAMES[method]}|{VARIANT_NAMES[variant]}|{BANK_NAMES[bank]}|{_cell(s,1)}|{_cell(s,3)}|{_cell(s,5)}|{s['mean_rank']:.2f}|")
    lines += ["", "## 本轮修改图相对原图的增量", "", "|方法|第一增量（百分点）|前三增量（百分点）|前五增量（百分点）|平均名次变化|前三新增请求|前三损失请求|", "|---|---:|---:|---:|---:|---:|---:|"]
    for method, result in methods.items():
        delta = result["modified_minus_clean"]["vague_joint"]
        paired = result["current_clean_modified_paired"]["top3"]
        lines.append(f"|{METHOD_NAMES[method]}|{delta['top1_percentage_point_delta']:+.2f}|{delta['top3_percentage_point_delta']:+.2f}|{delta['top5_percentage_point_delta']:+.2f}|{delta['mean_rank_delta']:+.2f}|{paired['gained']}|{paired['lost']}|")
    lines += ["", "平均名次变化为修改图减原图，负值表示提前；比例增量为修改库减原图库。", "", "## 九组逐项结果", "", "|方法|组别|查询条件|图库|第一|前三|前五|平均名次|", "|---|---|---|---|---:|---:|---:|---:|"]
    for method, result in methods.items():
        for group in result["groups"]:
            for variant in VARIANT_NAMES:
                for bank in BANK_NAMES:
                    s = group["metrics"][variant][bank]
                    lines.append(f"|{METHOD_NAMES[method]}|{group['name']}|{VARIANT_NAMES[variant]}|{BANK_NAMES[bank]}|{_cell(s,1)}|{_cell(s,3)}|{_cell(s,5)}|{s['mean_rank']:.2f}|")
    lines += ["", "## 比较与解释边界", "", *["- " + text for text in summary["limitations"]], "", "已逐项核对本轮与最近特点图文条件的记忆库字节、源记忆身份、冻结修改图、实际模型与解码器、候选图片和文字摘要、查询照片摘要一致。完整逐请求名次、同轮配对新增与损失、跨条件聚合差值及所有证据摘要保存在同目录汇总文件中。", "", "本轮只读取旧证据，没有购物、策略模型调用、重新生成或训练；模拟商品参考的来源标签保留。", ""]
    freeze_json(run / "SUMMARY.json", summary)
    _freeze_text(run / "REPORT.md", "\n".join(lines))
    completion_file = run / "COMPLETED.json"
    completed_at = read_json(completion_file)["completed_at"] if completion_file.exists() else datetime.now(timezone.utc).isoformat()
    freeze_json(completion_file, dict(status="completed", completed_at=completed_at, plan_sha256=plan_sha,
        rankings=ranking_count, score_comparisons=score_count, methods=list(METHOD_NAMES), shopping_calls=0, generation_calls=0,
        summary_sha256=_sha(run / "SUMMARY.json"), report_sha256=_sha(run / "REPORT.md"),
        independent_verification=certificates, input_files=inputs, summary_content_sha256=digest(summary)))
    return summary
