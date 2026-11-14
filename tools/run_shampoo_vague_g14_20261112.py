"""冻结模糊图文输入，仅洗发水组更换巨型评估编码器；不生成、不购物。"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / "tools")]
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                  TOKENIZERS_PARALLELISM="false", PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.visual.audit import read_json, save_json, freeze_json, digest
from src.visual.image_utils import sha256_file

RUN = PROJECT / "output/shampoo_vague_g14_20261112"
BASE = PROJECT / "output/groups9_vague_multimodal_20261111"
BASE_PLAN_SHA = "849951c38a3109deebcd318b50d242a4909c125aaca4ac0f68e208c2b9dbdd6c"
METHODS = ("lucid", "perlage_no_pixel")
GROUP = "07_shampoo"


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def check(path, expected):
    if sha256_file(path) != expected:
        raise ValueError("冻结输入摘要不符：" + str(path))


def update_status(state, **extra):
    save_json(RUN / "RUN_STATUS.json", dict(status=state, updated_at=now(), pid=os.getpid(), **extra))


def prepare():
    RUN.mkdir(parents=True, exist_ok=True)
    if (RUN / "EXPERIMENT_PLAN.json").exists():
        return validate()
    source = BASE / "SCHEDULED_INPUTS.json"
    check(source, BASE_PLAN_SHA)
    baseline = read_json(source)
    selected = {}
    inputs = {str(source): BASE_PLAN_SHA}

    def bind(path, expected=None):
        path = Path(path).resolve()
        actual = sha256_file(path)
        if expected is not None and actual != expected:
            raise ValueError("准备阶段输入不符：" + str(path))
        inputs[str(path)] = actual

    for method in METHODS:
        group = copy.deepcopy(next(g for g in baseline["methods"][method]["groups"] if g["key"] == GROUP))
        if len(group["queries"]) != 20:
            raise ValueError("洗发水请求不是二十条")
        for query in group["queries"]:
            if query["query_mode"] != "joint":
                raise ValueError("必须是原冻结图文查询")
            bind(query["query_image"], query["query_image_sha256"])
        bind(group["frozen_file"], group["frozen_file_sha256"])
        bind(group["frozen_image"], group["frozen_image_sha256"])
        for kind in ("clean", "modified"):
            bound = group["banks"][kind]
            bind(bound["path"], bound["sha256"])
            entries = read_json(bound["path"])
            if len(entries) != 189 or len({e["Id"] for e in entries}) != 189:
                raise ValueError("必须检索完整一百八十九条唯一记忆")
            for entry in entries:
                image = entry.get("RetrievalImage") or entry["PrimaryImage"]
                bind(image["LocalPath"], image["SHA256"])
            previous = BASE / "methods" / method / "retrieval" / GROUP / (kind + ".json")
            bind(previous)
        selected[method] = group
    old_model_evidence = PROJECT / "output/shampoo_k13_20261106/retrieval/RANKINGS.json"
    bind(old_model_evidence)
    old_spec = read_json(old_model_evidence)["protocol"]["encoder"]["surrogate"]
    config = dict(setting="held_out_public_encoder", surrogate={k: old_spec[k]
                  for k in ("name", "pretrained", "hf_repo", "weight_path")},
                  expected_weight_sha256=old_spec["weight_sha256"],
                  reason="user_requested_single_encoder_replacement_shampoo_only",
                  activation="native_pretrained_architecture", precision="fp32")
    if config["surrogate"]["name"] != "ViT-g-14":
        raise ValueError("本地身份不是指定巨型模型")
    check(config["surrogate"]["weight_path"], config["expected_weight_sha256"])
    freeze_json(RUN / "evaluation_encoder.json", config)
    bind(RUN / "evaluation_encoder.json")
    for path in [Path(__file__), PROJECT / "tools/shampoo_g14_encoder.py",
                 PROJECT / "src/visual/visual_retrieval.py", PROJECT / "src/visual/encoders/joint.py"]:
        bind(path)
    plan = dict(schema_version=1, created_at=now(), scope="single_group_encoder_intervention",
                baseline_plan=dict(path=str(source), sha256=BASE_PLAN_SHA),
                group=GROUP, methods=selected, condition="none", pool_count=189,
                queries_per_bank=20, expected_rankings=80, expected_score_checks=15120,
                encoder_config=str(RUN / "evaluation_encoder.json"),
                encoder_config_sha256=sha256_file(RUN / "evaluation_encoder.json"),
                key_image_weight=0.5, query_image_weight=0.5, index_text="product_feature",
                device="cuda", precision="fp32", training_enabled=False,
                generation_enabled=False, shopping_enabled=False,
                only_semantic_change="evaluation_encoder_B32_to_native_g14",
                input_sha256=inputs)
    freeze_json(RUN / "EXPERIMENT_PLAN.json", plan)
    update_status("prepared", rankings=0)
    return validate()


def validate():
    plan = read_json(RUN / "EXPERIMENT_PLAN.json")
    if (tuple(plan["methods"]) != METHODS or plan["group"] != GROUP
            or plan["pool_count"] != 189 or plan["expected_rankings"] != 80
            or plan["expected_score_checks"] != 15120 or plan["shopping_enabled"]
            or plan["generation_enabled"] or plan["training_enabled"]):
        raise ValueError("实验范围改变")
    for filename, expected in plan["input_sha256"].items():
        check(filename, expected)
    baseline = read_json(plan["baseline_plan"]["path"])
    for method in METHODS:
        old = next(g for g in baseline["methods"][method]["groups"] if g["key"] == GROUP)
        if plan["methods"][method] != old:
            raise ValueError("原冻结图片、查询、库或商品身份被改变")
    return plan


class MemoizedEncoder:
    """仅复用本进程新编码结果，不加载任何磁盘向量。"""
    supports_image_weight = True

    def __init__(self, base):
        self.base = base
        self.snapshot = base.snapshot
        self.cache = {}
        self.last_encoding_audit = None

    def encode_joint(self, text, image, image_weight=0.5):
        key = (text, sha256_file(image), image_weight)
        if key not in self.cache:
            vector = self.base.encode_joint(text, image, image_weight=image_weight)
            self.cache[key] = (vector, copy.deepcopy(self.base.last_encoding_audit))
        vector, audit = self.cache[key]
        self.last_encoding_audit = copy.deepcopy(audit)
        return vector


def new_encoder(plan):
    from shampoo_g14_encoder import make_encoder
    raw = make_encoder(plan["encoder_config"], device="cuda")
    actual = raw.snapshot["surrogate"]
    expected = read_json(plan["encoder_config"])
    if (actual["name"] != "ViT-g-14" or actual["device"] != "cuda"
            or actual["precision"] != "fp32"
            or actual["weight_sha256"] != expected["expected_weight_sha256"]
            or actual["loading"]["actual_activation_modules"] != {"QuickGELU": 0, "GELU": 64}):
        raw.close()
        raise ValueError("实际巨型编码器身份、原生激活或设备不符")
    return raw, MemoizedEncoder(raw)


def result_path(method, kind):
    return RUN / "methods" / method / "retrieval" / GROUP / (kind + ".json")


def ranker_for(encoder, group, kind):
    from src.visual.visual_retrieval import VisualMemoryRetriever, detect_index_text_policy
    bank_path = group["banks"][kind]["path"]
    bank = read_json(bank_path)
    if detect_index_text_policy(bank) != "product_feature":
        raise ValueError("真实特点文字键改变")
    return bank, VisualMemoryRetriever(encoder, bank_path, cache_dir=None,
                 query_mode="joint", index_text="product_feature",
                 key_image_weight=0.5, query_image_weight=0.5)


def check_result(result, plan, method, kind):
    group = plan["methods"][method]
    if (result["plan_sha256"] != sha256_file(RUN / "EXPERIMENT_PLAN.json")
            or len(result["rankings"]) != 20 or result["pool_count"] != 189):
        raise ValueError("结果绑定或数量不符")
    baseline = read_json(BASE / "methods" / method / "retrieval" / GROUP / (kind + ".json"))
    for query, row, old in zip(group["queries"], result["rankings"], baseline["rankings"]):
        if row["task_id"] != query["task_id"] or len(row["ranking"]) != 189:
            raise ValueError("结果未覆盖请求和全库")
        old_by_id = {x["Id"]: x for x in old["ranking"]}
        if len(old_by_id) != 189 or {x["Id"] for x in row["ranking"]} != set(old_by_id):
            raise ValueError("候选身份改变")
        for item in row["ranking"]:
            original = old_by_id[item["Id"]]
            for field in ("index", "image_sha256", "text_sha256", "query_text_sha256", "query_image_sha256", "bank_sha256"):
                if item[field] != original[field]:
                    raise ValueError("除编码器外的输入改变：" + field)
        source = next(x for x in row["ranking"] if x["Id"] == group["source_memory_id"])
        if source["rank"] != row["source_rank"]:
            raise ValueError("源名次提取错误")


def compute(stage):
    plan = validate()
    raw, encoder = new_encoder(plan)
    verification = dict(status="passed", fresh_process=True, fresh_encoder=True,
                        execution_device="cuda", no_disk_vector_cache_read=True,
                        score_comparisons=0, rankings_recomputed=0, max_score_error=0.0, files={})
    try:
        for method in METHODS:
            group = plan["methods"][method]
            for kind in ("clean", "modified"):
                update_status("running", stage=stage, method=method, bank=kind)
                bank, ranker = ranker_for(encoder, group, kind)
                path = result_path(method, kind)
                if stage == "retrieve" and path.exists():
                    check_result(read_json(path), plan, method, kind)
                    continue
                previous = read_json(path) if stage == "verify" else None
                if previous:
                    check_result(previous, plan, method, kind)
                    if previous["protocol"]["encoder"] != encoder.snapshot:
                        raise ValueError("独立复算实际编码器改变")
                rows = []
                for index, query in enumerate(group["queries"]):
                    actual = ranker.rank(bank, query["instruction"], query["query_image"])
                    if len(actual) != 189 or len({item["Id"] for item in actual}) != 189:
                        raise ValueError("实际排名未覆盖完整唯一候选")
                    source = next(x for x in actual if x["Id"] == group["source_memory_id"])
                    rows.append(dict(task_id=query["task_id"], source_rank=source["rank"], source_score=source["score"],
                                     selected_ids=[x["Id"] for x in actual[:3]], ranking=actual))
                    if previous:
                        for a, b in zip(actual, previous["rankings"][index]["ranking"]):
                            for field in ("Id", "rank", "index", "image_sha256", "text_sha256", "model_sha256",
                                          "query_text_sha256", "query_image_sha256", "bank_sha256"):
                                if a[field] != b[field]:
                                    raise ValueError("独立复算名次或身份不符：" + field)
                            error = abs(a["score"] - b["score"])
                            if not math.isfinite(error) or error > 1e-6:
                                raise ValueError("独立复算分数差异超过容差")
                            verification["max_score_error"] = max(verification["max_score_error"], error)
                            verification["score_comparisons"] += 1
                        verification["rankings_recomputed"] += 1
                if stage == "retrieve":
                    result = dict(method=method, group=GROUP, bank=kind, rankings=rows, pool_count=189,
                                  protocol=copy.deepcopy(ranker.snapshot), plan_sha256=sha256_file(RUN / "EXPERIMENT_PLAN.json"),
                                  fresh_encoder=True, no_disk_vector_cache_read=True,
                                  top1=sum(r["source_rank"] == 1 for r in rows),
                                  top3=sum(r["source_rank"] <= 3 for r in rows), top5=sum(r["source_rank"] <= 5 for r in rows))
                    check_result(result, plan, method, kind)
                    freeze_json(path, result)
                else:
                    verification["files"][str(path)] = sha256_file(path)
                print(json.dumps(dict(stage=stage, method=method, bank=kind, completed_queries=20,
                                      source_ranks=[r["source_rank"] for r in rows]), ensure_ascii=False), flush=True)
        validate()
        if stage == "verify":
            if (verification["score_comparisons"], verification["rankings_recomputed"]) != (15120, 80):
                raise ValueError("独立复算范围不完整")
            freeze_json(RUN / "INDEPENDENT_VERIFICATION.json", verification)
        else:
            freeze_json(RUN / "RETRIEVAL_COMPLETED.json", dict(status="completed", rankings=80,
                        plan_sha256=sha256_file(RUN / "EXPERIMENT_PLAN.json")))
    finally:
        encoder.cache.clear()
        raw.close()


def metrics(result):
    ranks = [row["source_rank"] for row in result["rankings"]]
    return dict(n=len(ranks), top1=sum(x == 1 for x in ranks), top3=sum(x <= 3 for x in ranks),
                top5=sum(x <= 5 for x in ranks), mean_rank=sum(ranks) / len(ranks),
                minimum_rank=min(ranks), maximum_rank=max(ranks), mrr=sum(1 / x for x in ranks) / len(ranks))


def report():
    plan = validate()
    v = read_json(RUN / "INDEPENDENT_VERIFICATION.json")
    if v["status"] != "passed" or v["rankings_recomputed"] != 80 or v["score_comparisons"] != 15120:
        raise ValueError("复核未完整通过")
    for path, expected in v["files"].items():
        check(path, expected)
    rows = []
    for method in METHODS:
        for kind in ("clean", "modified"):
            current = read_json(result_path(method, kind))
            previous = read_json(BASE / "methods" / method / "retrieval" / GROUP / (kind + ".json"))
            rows.append(dict(method=method, bank=kind, baseline=metrics(previous), g14=metrics(current)))
    freeze_json(RUN / "SUMMARY.json", dict(rows=rows, verification=v,
                only_changed="evaluation_encoder", generation_runs=0, shopping_runs=0))
    lines = ["公子，本轮仅洗发水组更换评估编码器，已完成无防御检索及独立复算。", "",
             "复用旧模糊图文轮二十条中文泛句、同一独立照片、两方法原有冻结图、完整一百八十九条原图库及修改库；仅将基础型三十二像素图像块编码器换成巨型十四像素图像块编码器。图文权重各半，记忆仍用真实特点文字键。", "",
             "|方法|图库|原编码器第一／前三／前五|巨型编码器第一／前三／前五|原平均名次|巨型平均名次|巨型名次范围|", "|---|---|---|---|---:|---:|---|"]
    for row in rows:
        a, b = row["baseline"], row["g14"]
        name = "原始像素方法" if row["method"] == "lucid" else "无像素感知方法"
        kind = "原图库" if row["bank"] == "clean" else "修改库"
        lines.append(f"|{name}|{kind}|{a['top1']}/20；{a['top3']}/20；{a['top5']}/20|{b['top1']}/20；{b['top3']}/20；{b['top5']}/20|{a['mean_rank']:.2f}|{b['mean_rank']:.2f}|{b['minimum_rank']}–{b['maximum_rank']}|")
    lines += ["", f"两个独立进程重新加载本地模型并全量编码；复算八十条排名、一万五千一百二十个候选分数，最大分数误差为{v['max_score_error']}。原始输入摘要在运行前后通过核验。", "",
              "实际模型使用原生普通激活、严格本地权重、显卡及单精度；未修改现有基础编码器或全局配置。模型身份及加载来源保存在逐条件协议。", "",
              "本轮只做检索，不重新生成、训练或购物。结果不能称为误购率；只隔离当前冻结输入下编码器替换的效果，不是旧高购物成功实验的完整复现。"]
    (RUN / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    freeze_json(RUN / "COMPLETED.json", dict(status="completed", finished_at=now(), rankings=80,
                score_comparisons=15120, report_sha256=sha256_file(RUN / "REPORT.md"),
                summary_sha256=sha256_file(RUN / "SUMMARY.json")))
    update_status("completed", rankings=80, score_comparisons=15120)


def queue_locked():
    prepare()
    for stage, marker in [("retrieve", "RETRIEVAL_COMPLETED.json"), ("verify", "INDEPENDENT_VERIFICATION.json")]:
        if (RUN / marker).exists():
            continue
        log = RUN / "logs" / (stage + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-X", "utf8", "-B", "-u", str(Path(__file__).resolve()), "--stage", stage]
        with log.open("a", encoding="utf-8") as handle:
            child = subprocess.Popen(command, cwd=PROJECT, stdout=handle, stderr=subprocess.STDOUT)
            save_json(RUN / "ACTIVE_CHILD.json", dict(pid=child.pid, stage=stage, command=command, launched_at=now()))
            print(json.dumps(dict(stage=stage, pid=child.pid, log=str(log))), flush=True)
            code = child.wait()
        if code != 0 or not (RUN / marker).exists():
            raise RuntimeError(f"阶段失败：{stage}，退出码{code}，证据已保存")
    report()
    print("洗发水巨型编码器检索及独立复算全部完成。", flush=True)


def queue():
    import msvcrt
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / "RUN.lock").open("a+b") as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"1")
            handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            queue_locked()
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["prepare", "queue", "retrieve", "verify", "report"], default="queue")
    args = parser.parse_args()
    try:
        if args.stage == "queue": queue()
        elif args.stage == "prepare": prepare()
        elif args.stage == "report": report()
        else: compute(args.stage)
    except Exception as exc:
        update_status("failed", stage=args.stage, error=repr(exc))
        save_json(RUN / "FAILURE.json", dict(stage=args.stage, failed_at=now(), error=repr(exc), traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
