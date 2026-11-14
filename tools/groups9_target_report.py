"""冻结九组特点文字加目标商品原图检索报告；只读取已完成排名与独立复核证据。"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import mean

from src.visual.audit import freeze_json, read_json


PROJECT = Path(__file__).resolve().parents[1]
METHOD_NAMES = {"lucid": "原始像素方法", "perlage_no_pixel": "无像素感知方法"}
BANK_NAMES = {"clean": "原图库", "modified": "修改库"}
CONDITION_NAMES = {
    "target_joint": "本次特点文字加目标原图",
    "feature_joint": "最近特点文字加独立照片",
    "feature_text": "最近特点纯文字",
    "vague_joint": "最近模糊文字加独立照片",
}
COMPARISON = PROJECT / "output/groups9_vague_multimodal_20261111/SUMMARY.json"
COMPARISON_COMPLETED = PROJECT / "output/groups9_vague_multimodal_20261111/COMPLETED.json"
FEATURE_SUMMARY = PROJECT / "output/groups9_multimodal_20261110/RETRIEVAL_SUMMARY.json"


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


def _aggregate(rows: list[list[int]]) -> dict:
    merged = [rank for group in rows for rank in group]
    return _stats(merged)


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
        ranks.append(actual["rank"])
    return ranks


def _cell(stats: dict, k: int) -> str:
    return f"{stats[f'top{k}']}/{stats['query_count']}（{stats[f'top{k}_rate']:.2%}）"


def _mean_cell(stats: dict) -> str:
    return f"{stats['mean_rank']:.2f}"


def write_report(run: Path, plan: dict) -> dict:
    run = Path(run).resolve()
    if not run.is_relative_to(PROJECT / "output"):
        raise ValueError("报告只能写入当前项目独立输出目录")
    plan_file = run / "SCHEDULED_INPUTS.json"
    plan_sha = _sha(plan_file)
    if read_json(plan_file) != plan or plan.get("shopping_enabled") is not False or plan.get("generation_enabled") is not False:
        raise ValueError("报告输入不是本轮冻结的只检索计划")
    if set(plan.get("methods", {})) != set(METHOD_NAMES):
        raise ValueError("报告必须包含指定的两种方法")
    if not COMPARISON_COMPLETED.is_file():
        raise ValueError("最近模糊图文轮完成标记缺失")
    comparison = read_json(COMPARISON)
    if comparison.get("status") != "completed" or comparison.get("rankings") != 720:
        raise ValueError("最近模糊图文汇总状态不符")
    feature_summary = read_json(FEATURE_SUMMARY)
    inputs = {str(plan_file): plan_sha, str(COMPARISON): _sha(COMPARISON),
              str(COMPARISON_COMPLETED): _sha(COMPARISON_COMPLETED),
              str(FEATURE_SUMMARY): _sha(FEATURE_SUMMARY)}

    methods = {}
    certificates = {}
    ranking_count = score_count = 0
    for method in METHOD_NAMES:
        groups = plan["methods"][method]["groups"]
        if len(groups) != 9 or len({g["key"] for g in groups}) != 9:
            raise ValueError("每方法必须包含九个不同商品组")
        cert_file = run / "methods" / method / "INDEPENDENT_VERIFICATION.json"
        cert = read_json(cert_file)
        if (cert.get("status") != "passed" or cert.get("score_comparisons") != 68040
                or cert.get("rankings_recomputed") != 360 or cert.get("plan_sha256") != plan_sha):
            raise ValueError("方法独立复核证书不完整或不属于本轮计划")
        if not math.isfinite(cert.get("max_score_error", math.inf)) or cert["max_score_error"] > 1e-6:
            raise ValueError("独立复核分数误差超过容差")
        for filename, expected in cert.get("files", {}).items():
            if _sha(Path(filename)) != expected:
                raise ValueError("独立复核绑定结果已改变：" + filename)
        certificates[method] = dict(path=str(cert_file), sha256=_sha(cert_file),
            score_comparisons=cert["score_comparisons"], rankings_recomputed=cert["rankings_recomputed"],
            max_score_error=cert["max_score_error"])
        score_count += cert["score_comparisons"]
        group_rows = []
        clean_all, modified_all = [], []
        for group in groups:
            per_bank = {}
            for bank_name in ("clean", "modified"):
                result = read_json(run / "methods" / method / "retrieval" / group["key"] / (bank_name + ".json"))
                ranks = _rank_rows(result, group["source_memory_id"], group["queries"],
                                   group["banks"][bank_name]["sha256"])
                per_bank[bank_name] = _stats(ranks)
                (clean_all if bank_name == "clean" else modified_all).append(ranks)
            group_rows.append(dict(key=group["key"], name=group["name"],
                clean=per_bank["clean"], modified=per_bank["modified"]))
        methods[method] = dict(groups=group_rows,
            clean=_aggregate(clean_all), modified=_aggregate(modified_all))
        ranking_count += 360
    if ranking_count != 720 or score_count != 136080:
        raise ValueError("总体证据范围不符")

    history = {}
    for method, raw in comparison["methods"].items():
        history[method] = {}
        for variant in CONDITION_NAMES:
            if variant == "target_joint":
                continue
            overall = raw.get("overall", {}).get(variant)
            if not overall or overall["clean"]["query_count"] != 180 or overall["modified"]["query_count"] != 180:
                raise ValueError("历史条件总体记录缺失或数量不符：" + method + " " + variant)
            history[method][variant] = {bank: _stats(overall[bank]["source_ranks"])
                                        for bank in ("clean", "modified")}
    for method in METHOD_NAMES:
        summary = feature_summary["methods"][method]
        if summary["overall"]["text_modified"]["top3"] != history[method]["feature_text"]["modified"]["top3"]:
            raise ValueError("纯文字基线两份证据前三不符：" + method)
        if summary["overall"]["modified"]["top3"] != history[method]["feature_joint"]["modified"]["top3"]:
            raise ValueError("特点图文两份证据前三不符：" + method)

    lines = ["# 九组特点文字加目标商品原图检索复测", "",
        "本轮仅无防御检索，原始像素方法与无像素感知方法串行执行。逐字复用十一月十日二十条特点文字，"
        "查询图片改为每组目标商品原图（含目标商品自身品牌印刷），这是用户于2026年11月11日明确指定的新条件，"
        "不改变项目默认的独立生成无品牌模糊照片协议。每次检索完整一百八十九条共享记忆。", "",
        "两方法独立复核全部通过：七百二十条全库排名、十三万六千零八十个候选分数。以下统计指定源记忆的名次，不能解释为最终误购率。", "",
        "## 总体结果与历史条件", "",
        "|方法|查询条件|图库|第一|前三|前五|平均名次|",
        "|---|---|---|---:|---:|---:|---:|"]
    for method in METHOD_NAMES:
        for variant in CONDITION_NAMES:
            for bank in ("clean", "modified"):
                if variant == "target_joint":
                    stats = methods[method][bank]
                else:
                    stats = history[method][variant][bank]
                lines.append(f"|{METHOD_NAMES[method]}|{CONDITION_NAMES[variant]}|{BANK_NAMES[bank]}|"
                    f"{_cell(stats, 1)}|{_cell(stats, 3)}|{_cell(stats, 5)}|{_mean_cell(stats)}|")
    lines += ["", "## 本轮修改图相对原图的增量", "",
        "|方法|第一增量（百分点）|前三增量（百分点）|前五增量（百分点）|平均名次变化|", "|---|---:|---:|---:|---:|"]
    for method in METHOD_NAMES:
        clean, modified = methods[method]["clean"], methods[method]["modified"]
        lines.append(f"|{METHOD_NAMES[method]}|"
            f"{100 * (modified['top1_rate'] - clean['top1_rate']):+.2f}|"
            f"{100 * (modified['top3_rate'] - clean['top3_rate']):+.2f}|"
            f"{100 * (modified['top5_rate'] - clean['top5_rate']):+.2f}|"
            f"{modified['mean_rank'] - clean['mean_rank']:+.2f}|")
    lines += ["", "## 本轮与最近特点文字加独立照片的修改库差值", "",
        "|方法|前三差值（百分点）|前五差值（百分点）|平均名次差值|", "|---|---:|---:|---:|"]
    for method in METHOD_NAMES:
        current = methods[method]["modified"]
        previous = history[method]["feature_joint"]["modified"]
        lines.append(f"|{METHOD_NAMES[method]}|"
            f"{100 * (current['top3_rate'] - previous['top3_rate']):+.2f}|"
            f"{100 * (current['top5_rate'] - previous['top5_rate']):+.2f}|"
            f"{current['mean_rank'] - previous['mean_rank']:+.2f}|")
    lines += ["", "## 九组逐项结果", "",
        "|方法|组别|图库|前三|前五|平均名次|", "|---|---|---|---:|---:|---:|"]
    for method in METHOD_NAMES:
        for group in methods[method]["groups"]:
            for bank in ("clean", "modified"):
                stats = group[bank]
                lines.append(f"|{METHOD_NAMES[method]}|{group['name']}|{BANK_NAMES[bank]}|"
                    f"{_cell(stats, 3)}|{_cell(stats, 5)}|{_mean_cell(stats)}|")
    lines += ["", "## 比较与解释边界", "",
        "- 本次只统计指定源记忆的检索名次，不是最终购物或误购率。",
        "- 查询图片为目标商品原图，含品牌印刷和可读文字；这是用户指定的新测试条件，项目默认无品牌独立生成照片协议不变。",
        "- 本次文字与最近特点文字、特点图文轮逐字相同，只改变查询图片；与最近模糊文字轮的语言和具体度不同。",
        "- 每组二十种措辞共用一张查询图，不能作为二十个独立视觉情境。",
        "- 指定评估编码器与训练代理重合，不声称独立留出黑盒迁移；第四、八组保留历史选图门槛未通过状态。",
        "- 每组正式文字与实际训练正文有一条精确重合，须如实报告。",
        "- 本轮只执行检索与独立复算，没有购物、策略模型调用、重新生成或训练；模拟商品参考的来源标签保留。", ""]
    report_text = "\n".join(lines)
    summary = dict(schema_version=1, status="completed", scope="retrieval_only_no_defense",
        plan_sha256=plan_sha, rankings=720, independent_score_checks=136080,
        query_protocol=plan["query_protocol"], memory_protocol=plan["memory_protocol"],
        methods={method: dict(clean=methods[method]["clean"], modified=methods[method]["modified"],
            groups=methods[method]["groups"]) for method in METHOD_NAMES},
        history={method: {variant: {bank: history[method][variant][bank]
                    for bank in ("clean", "modified")} for variant in CONDITION_NAMES if variant != "target_joint"}
                 for method in METHOD_NAMES},
        certificates=certificates, inputs=inputs,
        scope_amendment=plan["scope_amendment"])
    freeze_json(run / "SUMMARY.json", summary)
    report_path = run / "REPORT.md"
    if report_path.exists():
        if report_path.read_text(encoding="utf-8") != report_text:
            raise ValueError("已有冻结报告内容不同，拒绝覆盖")
    else:
        temporary = report_path.with_suffix(".tmp")
        temporary.write_text(report_text, encoding="utf-8")
        temporary.replace(report_path)
    freeze_json(run / "COMPLETED.json", dict(status="completed", updated_at=plan.get("output", run),
        rankings=720, independent_score_checks=136080, plan_sha256=plan_sha,
        report_sha256=_sha(report_path), summary_sha256=_sha(run / "SUMMARY.json")))
    return summary
