import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

import numpy as np
from PIL import Image
import torch
from src.paths import PROJECT, OUTPUT, ProjectPath as PPath
from src.visual.optimization import generate_attack
from src.visual.catalog_training_guard import catalog_training_guard
from src.visual.audit import read_json, freeze_json
from src.visual.image_utils import sha256_file
from src.visual.encoders.clip import make_encoder
from src.visual.visual_retrieval import resolve_image, unit

print("=== Starting lucid 1000-step generation under pure QuickGELU ===", flush=True)

root = OUTPUT / "shampoo_lucid_quickgelu_16"
root.mkdir(parents=True, exist_ok=True)

data_dir = OUTPUT / "shampoo_ablation_20261107/data/07_shampoo"
source_img = data_dir / "source.jpg"
target_img = data_dir / "target.jpg"
historical_lucid = OUTPUT / "shampoo_ablation_20261107/images/lucid_original_16.png"

# 1. Run generation if not already completed
adversarial_img = root / "adversarial.png"
completion_file = root / "completion.json"

if not (adversarial_img.exists() and completion_file.exists()):
    start_time = time.perf_counter()
    with catalog_training_guard(allowed_output_roots=[root, data_dir]) as audit:
        image, metadata = generate_attack(
            source=source_img,
            target=target_img,
            output_dir=root,
            method="lucid",
            steps=1000,
            epsilon_levels=16,
            device="cuda",
        )
        freeze_json(completion_file, metadata)
    audit_clean = {k: v for k, v in audit.items() if k != "active"}
    freeze_json(root / "TRAINING_ACCESS_AUDIT.json", audit_clean)
    elapsed = time.perf_counter() - start_time
    print(f"Generation completed in {elapsed:.2f} s", flush=True)
else:
    print("Using existing generated adversarial.png", flush=True)
    metadata = read_json(completion_file)

print("=== Generation phase verified ===", flush=True)

# 2. Compute image-level similarity metrics
print("=== Computing pixel similarity metrics ===", flush=True)
src_arr = np.array(Image.open(source_img).convert("RGB"), dtype=np.float32)
new_arr = np.array(Image.open(adversarial_img).convert("RGB"), dtype=np.float32)
old_arr = np.array(Image.open(historical_lucid).convert("RGB"), dtype=np.float32)

from skimage.metrics import structural_similarity as ssim

new_ssim = float(ssim(src_arr, new_arr, channel_axis=2, data_range=255))
old_ssim = float(ssim(src_arr, old_arr, channel_axis=2, data_range=255))

new_mae = float(np.mean(np.abs(new_arr - src_arr)))
old_mae = float(np.mean(np.abs(old_arr - src_arr)))

new_linf = int(np.max(np.abs(new_arr - src_arr)))
old_linf = int(np.max(np.abs(old_arr - src_arr)))

print(f"New lucid (QuickGELU): SSIM={new_ssim:.4f}, MAE={new_mae:.2f}, Linf={new_linf}")
print(f"Old lucid (GELU):      SSIM={old_ssim:.4f}, MAE={old_mae:.2f}, Linf={old_linf}")

# 3. Evaluate Cosine similarities with QuickGELU ViT-B-32
print("=== Computing QuickGELU CLIP cosine similarities ===", flush=True)
encoder = make_encoder(surrogate=2, device="cuda")

def get_feat(path):
    return unit(encoder.encode(path))

src_feat = get_feat(source_img)
tgt_feat = get_feat(target_img)
new_feat = get_feat(adversarial_img)
old_feat = get_feat(historical_lucid)

new_cos_target = float(new_feat @ tgt_feat)
old_cos_target = float(old_feat @ tgt_feat)
src_cos_target = float(src_feat @ tgt_feat)

new_cos_source = float(new_feat @ src_feat)
old_cos_source = float(old_feat @ src_feat)

print(f"Clean source cosine to target:       {src_cos_target:.4f}")
print(f"New lucid (QuickGELU) to target:    {new_cos_target:.4f}")
print(f"Old lucid (GELU) to target:         {old_cos_target:.4f}")

# 4. Evaluate in 189-item shared memory pool
print("=== Evaluating 189-item shared memory retrieval ===", flush=True)
plan_path = OUTPUT / "shampoo_ablation_20261107/image_only/PLAN.json"
plan = read_json(plan_path)
bank = read_json(OUTPUT / "shampoo_ablation_20261107/image_only/banks/clean.json")
source_idx = next(i for i, e in enumerate(bank) if e["Id"] == plan["source_memory_id"])

# Compute vectors for all 189 entries
bank_vectors = np.stack([get_feat(resolve_image(e, OUTPUT / "shampoo_ablation_20261107/image_only/banks/clean.json")) for e in bank])
target_query_feat = tgt_feat

# Test new image in pool
bank_new = bank_vectors.copy()
bank_new[source_idx] = new_feat
scores_new = [float(x @ target_query_feat) for x in bank_new]
rank_new = sorted(range(len(scores_new)), key=lambda i: (-scores_new[i], i)).index(source_idx) + 1

# Test old image in pool
bank_old = bank_vectors.copy()
bank_old[source_idx] = old_feat
scores_old = [float(x @ target_query_feat) for x in bank_old]
rank_old = sorted(range(len(scores_old)), key=lambda i: (-scores_old[i], i)).index(source_idx) + 1

# Test clean source in pool
scores_clean = [float(x @ target_query_feat) for x in bank_vectors]
rank_clean = sorted(range(len(scores_clean)), key=lambda i: (-scores_clean[i], i)).index(source_idx) + 1

print(f"Clean source rank:       #{rank_clean} (score={scores_clean[source_idx]:.4f})")
print(f"New lucid rank (target): #{rank_new} (score={scores_new[source_idx]:.4f})")
print(f"Old lucid rank (target): #{rank_old} (score={scores_old[source_idx]:.4f})")

# 5. Multimodal retrieval with 20 formal requests
print("=== Evaluating multimodal retrieval across 20 formal requests ===", flush=True)
from src.experiments.shared_evaluation import make_eval_encoder
from src.visual.visual_retrieval import resolve_image
from src.experiments.formal_queries import VAGUE_INSTRUCTIONS
from src.agent.memory.balanced_pool import validate_balanced_pool

joint = make_eval_encoder('cuda')
query_img_path = OUTPUT / "shampoo_ablation_20261107/evaluation_inputs/query.png"

# Cache bank entry joint vectors (188 non-source entries)
raw_bank = validate_balanced_pool(OUTPUT / "shared_memory/pool_189_v1/shared_memory.json")
source_id = plan["source_memory_id"]
source_entry = next(e for e in raw_bank if e["Id"] == source_id)
other_entries = [e for e in raw_bank if e["Id"] != source_id]

# Encode queries
query_vectors = [joint.encode_joint(q, query_img_path, image_weight=0.5) for q in VAGUE_INSTRUCTIONS]

# Encode other 188 bank entries
other_vectors = []
for e in other_entries:
    img_p = resolve_image(e, OUTPUT / "shared_memory/pool_189_v1/shared_memory.json")
    other_vectors.append(joint.encode_joint(e.get("Title") or e.get("ItemName", ""), img_p, image_weight=0.5))
other_vectors = np.stack(other_vectors)

def evaluate_variant(img_path):
    # Encode the modified source entry
    src_vec = joint.encode_joint(source_entry.get("Title") or source_entry.get("ItemName", ""), img_path, image_weight=0.5)
    all_vecs = np.vstack([src_vec, other_vectors]) # index 0 is source
    ranks = []
    scores = []
    for q_vec in query_vectors:
        s = all_vecs @ q_vec
        scores.append(float(s[0]))
        r = sorted(range(len(s)), key=lambda i: (-s[i], i)).index(0) + 1
        ranks.append(r)
    return {
        "mean_rank": float(np.mean(ranks)),
        "ranks": ranks,
        "top1": sum(1 for r in ranks if r <= 1),
        "top3": sum(1 for r in ranks if r <= 3),
        "top5": sum(1 for r in ranks if r <= 5),
        "mean_score": float(np.mean(scores))
    }

mm_new = evaluate_variant(adversarial_img)
mm_old = evaluate_variant(historical_lucid)
mm_clean = evaluate_variant(source_img)

print(f"Multimodal New lucid (QuickGELU): Mean Rank={mm_new['mean_rank']:.2f}, Top-3={mm_new['top3']}/20, Top-5={mm_new['top5']}/20")
print(f"Multimodal Old lucid (GELU):      Mean Rank={mm_old['mean_rank']:.2f}, Top-3={mm_old['top3']}/20, Top-5={mm_old['top5']}/20")
print(f"Multimodal Clean Source:          Mean Rank={mm_clean['mean_rank']:.2f}, Top-3={mm_clean['top3']}/20, Top-5={mm_clean['top5']}/20")

results = {
    "metrics": {
        "new_lucid_quickgelu_16": {
            "ssim": new_ssim,
            "mae": new_mae,
            "linf": new_linf,
            "cosine_to_target": new_cos_target,
            "cosine_to_source": new_cos_source,
            "rank_in_189_target_query": rank_new,
            "score_in_189_target_query": scores_new[source_idx],
            "multimodal_20_queries": mm_new
        },
        "old_lucid_gelu_16": {
            "ssim": old_ssim,
            "mae": old_mae,
            "linf": old_linf,
            "cosine_to_target": old_cos_target,
            "cosine_to_source": old_cos_source,
            "rank_in_189_target_query": rank_old,
            "score_in_189_target_query": scores_old[source_idx],
            "multimodal_20_queries": mm_old
        },
        "clean_source": {
            "ssim": 1.0,
            "mae": 0.0,
            "linf": 0,
            "cosine_to_target": src_cos_target,
            "cosine_to_source": 1.0,
            "rank_in_189_target_query": rank_clean,
            "score_in_189_target_query": scores_clean[source_idx],
            "multimodal_20_queries": mm_clean
        }
    },
    "delta": {
        "ssim_diff": new_ssim - old_ssim,
        "cosine_target_diff": new_cos_target - old_cos_target,
        "rank_diff": rank_new - rank_old,
        "mm_rank_diff": mm_new["mean_rank"] - mm_old["mean_rank"]
    },
    "perlage_decision": "NO_PERLAGE_REQUIRED" if rank_new == rank_old == 1 and mm_new["top3"] == mm_old["top3"] == 20 else "PERLAGE_REQUIRED"
}

freeze_json(root / "COMPARISON_RESULTS.json", results)
print("=== Evaluation complete, results saved to COMPARISON_RESULTS.json ===", flush=True)
print(json.dumps(results, indent=2))
