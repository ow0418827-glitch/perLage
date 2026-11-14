"""只读汇总已有洗发水轨迹，不导入实验模块，不调用模型或购物。"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
LEGACY = ROOT.parent / "lage"
SOURCE = "B08XYBS678"
TARGET = "B00BDJODWI"
INPUTS = {}


def read(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    INPUTS[str(path)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw.decode("utf-8-sig"))


def image_hash(path):
    path = Path(path).resolve()
    value = hashlib.sha256(path.read_bytes()).hexdigest()
    INPUTS[str(path)] = value
    return value


def rank_stats(ranks):
    return {"n": len(ranks), "min": min(ranks), "max": max(ranks),
            "mean": mean(ranks), "top1": sum(r == 1 for r in ranks),
            "top3": sum(r <= 3 for r in ranks), "top5": sum(r <= 5 for r in ranks)}


def summarize(records):
    rows = []
    for origin, record in records:
        episode = record.get("trajectory", record)
        trace = episode.get("PolicyTrace", [])
        source_flags = [any(m.get("ASIN") == SOURCE for m in step.get("MemoryImages", []))
                        for step in trace]
        query_hashes = sorted({m.get("SHA256") for step in trace
                               for m in step.get("InputImages", [])
                               if m.get("Role") == "query" and m.get("SHA256")})
        source_ids = {m.get("Id") for step in trace for m in step.get("MemoryImages", [])
                      if m.get("ASIN") == SOURCE}
        source_ids.update({"fixed_nine_p7_history_0_None",
                           "shared_759623cdb6bb5755f91a445b0e1bf4635684824e214fa1b190e23413d6814831",
                           "groups9_reference_07_shampoo_B08XYBS678"})
        retrieval_trace = episode.get("RetrievalTrace", [])
        source_rank = next((m.get("rank", i + 1)
                            for i, m in enumerate(retrieval_trace[0].get("Ranking", []))
                            if m.get("Id") in source_ids), None) if retrieval_trace else None
        row = {
            "origin": origin, "id": episode.get("Id", record.get("task_id")),
            "instruction": episode.get("Instruction", record.get("instruction")),
            "completed": episode.get("Completed", record.get("completed")),
            "final_asin": episode.get("FinalASIN", record.get("final_asin")),
            "reward": episode.get("EnvironmentReward", episode.get("Reward")),
            "termination": episode.get("TerminationReason"),
            "policy_steps": len(trace),
            "source_exposed_any_step": any(source_flags),
            "source_exposed_every_step": bool(source_flags) and all(source_flags),
            "source_exposure_steps": sum(source_flags),
            "memory_counts": sorted({len(t.get("MemoryImages", [])) for t in trace}),
            "query_image_hashes": query_hashes,
            "first_retrieval_source_rank": source_rank,
            "guidance_texts": sorted({t.get("AuxiliaryPrompt", "") for t in trace}),
            "models": sorted({t.get("PolicyRequest", {}).get("Model", "") for t in trace}),
            "temperatures": sorted({t.get("PolicyRequest", {}).get("Temperature") for t in trace
                                    if t.get("PolicyRequest", {}).get("Temperature") is not None}),
            "max_output_tokens": sorted({t.get("PolicyRequest", {}).get("MaxTokens") for t in trace
                                         if t.get("PolicyRequest", {}).get("MaxTokens") is not None}),
            "first_step_memory_asins": [m.get("ASIN") for m in trace[0].get("MemoryImages", [])] if trace else [],
            "first_search_action": next((a for a in episode.get("Actions", [])
                                          if a.startswith("Action: search[")), None),
            "source_id_in_actions_or_observations": SOURCE in "\n".join(episode.get("Actions", [])),
        }
        rows.append(row)
    values = [r["reward"] for r in rows if isinstance(r["reward"], (int, float))]
    ranks = [r["first_retrieval_source_rank"] for r in rows if r["first_retrieval_source_rank"] is not None]
    return {"n": len(rows), "completed": sum(bool(r["completed"]) for r in rows),
            "source_purchases": sum(r["final_asin"] == SOURCE for r in rows),
            "target_purchases": sum(r["final_asin"] == TARGET for r in rows),
            "source_exposed_episodes": sum(r["source_exposed_any_step"] for r in rows),
            "policy_steps": sum(r["policy_steps"] for r in rows),
            "source_exposure_steps": sum(r["source_exposure_steps"] for r in rows),
            "actual_first_retrieval_source_ranks": rank_stats(ranks) if ranks else None,
            "mean_environment_reward": mean(values) if values else None,
            "final_asin_counts": dict(Counter(r["final_asin"] for r in rows)),
            "rows": rows}


def folder_records(folder):
    return [(str(path.resolve()), read(path)) for path in sorted(folder.glob("*.json"))]


old_root = LEGACY / "src/output/lage_runs/webshop_vground_shampoo_hidden_label_20261104"
series = {}
for label in ["top1", "top3"]:
    path = old_root / label / "results.json"
    records = [(f"{path.resolve()}#index={i}", row) for i, row in enumerate(read(path))]
    series[f"20261104_{label}"] = summarize(records)

bridge_root = ROOT / "output/shampoo_k13_20261106"
for label in ["top1", "top3"]:
    series[f"20261106_{label}"] = summarize(folder_records(bridge_root / "shopping/episodes" / label))

for name, folder in {
    "current_no_pixel": "perlage_no_pixel_groups9_20261109",
    "current_lucid": "lucid_groups9_20261109",
    "current_clean_shopping_control": "lucid_groups9_clean_shopping_20261110",
}.items():
    series[name] = summarize(folder_records(ROOT / "output" / folder / "shopping/episodes/07_shampoo"))

current_dir = ROOT / "output/perlage_no_pixel_groups9_20261109/groups/07_shampoo/methods/PerLAGE_no_pixel"
retrieval = {}
for method, folder in [("no_pixel", current_dir),
                       ("lucid", ROOT / "output/lucid_groups9_20261109/groups/07_shampoo/methods/lucid")]:
    for row in read(folder / "evaluation/evaluation.json"):
        retrieval[f"current_{method}_{row['bank']}"] = rank_stats([q["rank"] for q in row["queries"]])

retrieval["20261106_modified"] = rank_stats([q["source_rank"] for q in
                                            read(bridge_root / "retrieval/RANKINGS.json")["rows"]])
for row in read(ROOT / "output/shampoo_ablation_20261107/evaluation/RANKINGS.json")["rows"]:
    retrieval[f"20261107_{row['variant']}"] = rank_stats([q["source_rank"] for q in row["queries"]])

frozen = read(current_dir / "FROZEN_IMAGE.json")
bridge_plan = read(bridge_root / "PLAN.json")
hashes = {
    "old_actual_image": image_hash(bridge_root / "inputs/07_shampoo/modified.png"),
    "current_actual_image": image_hash(current_dir / "frozen.png"),
    "old_query_image": image_hash(bridge_root / "inputs/07_shampoo/query.png"),
    "current_source": image_hash(ROOT / "output/experiment_base_groups9_20261109/groups/07_shampoo/source.jpg"),
    "current_target": image_hash(ROOT / "output/experiment_base_groups9_20261109/groups/07_shampoo/target.jpg"),
}
assert hashes["old_actual_image"] == bridge_plan["modified_image_sha256"]
assert hashes["current_actual_image"] == frozen["image_sha256"]

def source_record(path):
    obj = read(path)
    entries = obj if isinstance(obj, list) else obj["entries"]
    record = next(e for e in entries if e.get("FinalASIN") == SOURCE)
    return {"pool_count": len(entries), "Id": record.get("Id"),
            "RecordKind": record.get("RecordKind"), "ObservedPurchase": record.get("ObservedPurchase"),
            "Completed": record.get("Completed"), "Success": record.get("Success"),
            "actions_count": len(record.get("Actions", [])),
            "title": record.get("PrimaryImage", {}).get("Title"),
            "FeatureText": record.get("FeatureText"), "Attributes": record.get("Attributes")}

source_records = {
    "20261104": source_record(old_root / "07_shampoo_bank_none.json"),
    "20261106": source_record(bridge_root / "shopping/policy_bank_modified.json"),
    "current": source_record(ROOT / "output/perlage_no_pixel_groups9_20261109/shopping/policy_banks/07_shampoo_modified.json"),
}
training = read(current_dir / "generation/latent_generation_refinement/public_title_encoding_audit.json")
verification = {
    "current_no_pixel": read(current_dir / "evaluation/independent_verification.json"),
    "20261106": read(bridge_root / "retrieval/INDEPENDENT_VERIFICATION.json"),
    "20261107": read(ROOT / "output/shampoo_ablation_20261107/evaluation/INDEPENDENT_VERIFICATION.json"),
}
for origin, digest in INPUTS.items():
    assert hashlib.sha256(Path(origin).read_bytes()).hexdigest() == digest, origin

audit = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
         "scope": "saved_artifacts_only_no_model_or_shopping_calls",
         "generation_calls": 0, "encoder_calls": 0, "shopping_calls": 0,
         "formal_outputs_modified": False, "legacy_project_read_only": True,
         "series": series, "retrieval": retrieval, "image_hashes": hashes,
         "source_records": source_records, "current_training_encoding": training,
         "saved_independent_verification": verification,
         "input_sha256": INPUTS, "inputs_unchanged_after_analysis": True,
         "limits": ["Historical versus current protocols change several factors simultaneously.",
                    "Reported source purchases are not a causal increment over clean shopping.",
                    "No new intervention, generation, ranking or shopping was performed."]}
(OUT / "AUDIT.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"series": {k: {x: y for x, y in v.items() if x != "rows"}
                              for k, v in series.items()},
                  "retrieval": retrieval, "source_records": source_records,
                  "input_files_verified": len(INPUTS)}, ensure_ascii=False, indent=2))
