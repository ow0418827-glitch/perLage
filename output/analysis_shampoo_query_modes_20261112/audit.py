"""只读取保存的排名，核对洗发水四种查询条件；不加载编码器。"""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
INPUTS = {}


def read(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    INPUTS[str(path)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw.decode("utf-8-sig"))


def file_hash(path):
    path = Path(path).resolve()
    value = hashlib.sha256(path.read_bytes()).hexdigest()
    INPUTS[str(path)] = value
    return value


def stats(values):
    return dict(n=len(values), minimum=min(values), maximum=max(values), mean=mean(values),
                top1=sum(v == 1 for v in values), top3=sum(v <= 3 for v in values),
                top5=sum(v <= 5 for v in values))


runs = {"B_feature_photo": "groups9_multimodal_20261110",
        "C_vague_photo": "groups9_vague_multimodal_20261111",
        "D_feature_target": "groups9_target_image_20261111"}
plans = {mode: read(ROOT / "output" / run /
                   ("EXPERIMENT_PLAN.json" if mode.startswith("B_") else "SCHEDULED_INPUTS.json"))
         for mode, run in runs.items()}
groups = {mode: {method: next(g for g in plan["methods"][method]["groups"] if g["key"] == "07_shampoo")
                 for method in ["lucid", "perlage_no_pixel"]} for mode, plan in plans.items()}

results = {}
protocols = {}
first_queries = {}
for method, folder, identity in [("perlage_no_pixel", "perlage_no_pixel_groups9_20261109", "PerLAGE_no_pixel"),
                                 ("lucid", "lucid_groups9_20261109", "lucid")]:
    results[method] = {"A_feature_text": {}}
    for row in read(ROOT / "output" / folder / "groups/07_shampoo/methods" / identity / "evaluation/evaluation.json"):
        results[method]["A_feature_text"][row["bank"]] = {
            "stats": stats([q["rank"] for q in row["queries"]]), "rankings": row["queries"]}
    for mode, run in runs.items():
        group = groups[mode][method]
        results[method][mode] = {}
        for bank in ["clean", "modified"]:
            obj = read(ROOT / "output" / run / "methods" / method / "retrieval/07_shampoo" / (bank + ".json"))
            pool = read(group["banks"][bank]["path"])
            assert len(pool) == 189
            assert file_hash(group["banks"][bank]["path"]) == group["banks"][bank]["sha256"]
            by_id = {e["Id"]: e for e in pool}
            rows = obj["rankings"]
            assert len(rows) == 20
            source_text_hashes = set()
            reduced = []
            for row in rows:
                source = next(r for r in row["ranking"] if r["Id"] == group["source_memory_id"])
                assert source["rank"] == row["source_rank"]
                source_text_hashes.add(source["text_sha256"])
                reduced.append(dict(task_id=row["task_id"], source_rank=row["source_rank"],
                                    source_score=row["source_score"],
                                    source_minus_third=row["source_score"] - row["ranking"][2]["score"]))
            first_top = [dict(rank=r["rank"], asin=by_id[r["Id"]]["FinalASIN"],
                              title=by_id[r["Id"]]["PrimaryImage"]["Title"], score=r["score"])
                         for r in rows[0]["ranking"][:10]]
            results[method][mode][bank] = dict(stats=stats([r["source_rank"] for r in rows]),
                rankings=reduced, source_text_sha256=sorted(source_text_hashes),
                mean_source_minus_third=mean(r["source_minus_third"] for r in reduced),
                first_query_top10=first_top,
                top3_asin_frequencies=dict(Counter(by_id[r["Id"]]["FinalASIN"]
                                                  for row in rows for r in row["ranking"][:3])))
            protocols[(method, mode, bank)] = obj["protocol"]
        first_queries[f"{method}_{mode}"] = group["queries"][0]

comparisons = {}
for method in ["lucid", "perlage_no_pixel"]:
    b, c, d = (groups[k][method] for k in runs)
    stable_fields = ["source", "target", "source_memory_id", "banks", "frozen_image", "frozen_image_sha256"]
    stable = all(b[field] == c[field] == d[field] for field in stable_fields)
    assert stable
    b_text = [q["instruction"] for q in b["queries"]]
    c_text = [q["instruction"] for q in c["queries"]]
    d_text = [q["instruction"] for q in d["queries"]]
    assert b_text == d_text
    old = read(ROOT / "output/shampoo_k13_20261106/PLAN.json")
    assert c_text == [q["instruction"] for q in old["queries"]]
    assert [q["query_image_sha256"] for q in b["queries"]] == [q["query_image_sha256"] for q in c["queries"]]
    stripped = []
    for mode in runs:
        p = deepcopy(protocols[(method, mode, "modified")])
        p.pop("query_encoding_audit", None)
        stripped.append(p)
    protocol_identical = stripped[0] == stripped[1] == stripped[2]
    assert protocol_identical
    comparisons[method] = dict(stable_group_fields=stable,
        B_C_query_images_identical=True, B_D_query_texts_identical=True,
        C_texts_equal_old_20_vague_instructions=True,
        protocols_identical_except_query_text_encoding_audit=protocol_identical,
        mean_rank_improvement_by_modification={mode:
            results[method][mode]["clean"]["stats"]["mean"] - results[method][mode]["modified"]["stats"]["mean"]
            for mode in results[method]})

image_paths = {
    "new_independent_photo": groups["B_feature_photo"]["perlage_no_pixel"]["queries"][0]["query_image"],
    "old_label_removed_photo": ROOT / "output/shampoo_k13_20261106/inputs/07_shampoo/query.png",
    "current_no_pixel_frozen": groups["B_feature_photo"]["perlage_no_pixel"]["frozen_image"],
    "target_original": groups["B_feature_photo"]["perlage_no_pixel"]["target"]["image"],
}
images = {name: dict(path=str(Path(path).resolve()), sha256=file_hash(path)) for name, path in image_paths.items()}
verification = {}
for mode, run in runs.items():
    for method in ["lucid", "perlage_no_pixel"]:
        v = read(ROOT / "output" / run / "methods" / method / "INDEPENDENT_VERIFICATION.json")
        verification[f"{mode}_{method}"] = {k: v[k] for k in
            ["status", "max_score_error", "score_comparisons", "rankings_recomputed"] if k in v}

for path, digest in INPUTS.items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
out = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
           scope="saved_scores_and_frozen_inputs_only", encoder_calls=0, generation_calls=0, shopping_calls=0,
           results=results, controlled_comparisons=comparisons, query_images=images,
           first_queries=first_queries, saved_independent_verification=verification,
           input_sha256=INPUTS, inputs_unchanged_after_analysis=True,
           limits=["B to C changes both language and semantic specificity.",
                   "C and D are retrieval-only, not shopping outcomes.",
                   "The old label-removed photo has not been tested in an otherwise identical current configuration."])
(OUT / "AUDIT.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(dict(stats={method: {mode: {bank: value["stats"] for bank, value in banks.items()}
                                    for mode, banks in modes.items()} for method, modes in results.items()},
                      comparisons=comparisons, input_files=len(INPUTS)), ensure_ascii=True, indent=2))
