import os
import sys
from pathlib import Path

PROJECT_ROOT = Path("d:/Material/Literature Reading/code/perLage").resolve()
os.chdir(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

import copy
import gc
import json
import statistics
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

from src.paths import SHARED_MEMORY
from src.visual.audit import freeze_json, read_json
from src.visual.image_utils import sha256_file
from src.agent.memory.balanced_pool import validate_balanced_pool
from src.experiments.shared_evaluation import make_eval_encoder
from src.visual.encoders.clip import make_encoder
from src.visual.visual_retrieval import VisualMemoryRetriever, resolve_image
from src.experiments.shampoo_ablation_evaluation import MemoizedEncoder

def run():
    print("=== Running Official-Protocol Evaluation for Lucid under QuickGELU ===", flush=True)
    
    plan_path = PROJECT_ROOT / "output/shampoo_ablation_20261107/evaluation/PLAN.json"
    plan = read_json(plan_path)
    
    source_img_path = Path(plan["source_image"])
    target_img_path = Path(plan["target_image"])
    query_img_path = Path(plan["query_image"])
    
    historical_lucid_path = PROJECT_ROOT / "output/shampoo_ablation_20261107/images/lucid_original_16.png"
    new_lucid_path = PROJECT_ROOT / "output/shampoo_lucid_quickgelu_16/adversarial.png"
    
    variants = {
        "clean": {
            "name": "未修改原图",
            "image": str(source_img_path),
            "sha256": sha256_file(source_img_path)
        },
        "lucid_original_16": {
            "name": "原始像素方法旧图（十六级）",
            "image": str(historical_lucid_path),
            "sha256": sha256_file(historical_lucid_path)
        },
        "lucid_quickgelu_16": {
            "name": "原始像素方法新图（QuickGELU 十六级）",
            "image": str(new_lucid_path),
            "sha256": sha256_file(new_lucid_path)
        }
    }
    
    # 1. Similarities
    print("\n--- 1. Computing Image Similarities and Feature Cosines ---", flush=True)
    with Image.open(source_img_path) as im:
        original = np.asarray(im.convert("RGB")).copy()
        
    sim_rows = []
    for var_id, item in variants.items():
        with Image.open(item["image"]) as im:
            pixels = np.asarray(im.convert("RGB")).copy()
        error = np.abs(pixels.astype(float) - original.astype(float))
        psnr = peak_signal_noise_ratio(original, pixels, data_range=255) if error.max() else None
        sim_rows.append({
            "variant": var_id,
            "name": item["name"],
            "source_ssim": float(structural_similarity(original, pixels, channel_axis=2, data_range=255)),
            "source_mae_levels": float(error.mean()),
            "source_max_levels": int(error.max()),
            "source_psnr_db": float(psnr) if psnr is not None else None,
            "image_sha256": item["sha256"]
        })
        
    for label, factory in [('training_proxy_2', lambda: make_encoder(surrogate=2, device='cuda')),
                          ('held_out', lambda: make_eval_encoder('cuda'))]:
        encoder = factory()
        try:
            target_vec = encoder.encode(str(target_img_path))
            source_vec = encoder.encode(str(source_img_path))
            query_vec = encoder.encode(str(query_img_path))
            for row in sim_rows:
                vec = encoder.encode(variants[row["variant"]]["image"])
                row[label] = {
                    "target_image_cosine": float(vec @ target_vec),
                    "source_image_cosine": float(vec @ source_vec),
                    "query_image_cosine": float(vec @ query_vec),
                }
        finally:
            if hasattr(encoder, "close"):
                encoder.close()
            del encoder
            gc.collect()

    print("\nSimilarities summary:")
    for row in sim_rows:
        print(f"[{row['name']}] SSIM: {row['source_ssim']:.6f}, MAE: {row['source_mae_levels']:.4f}, Linf: {row['source_max_levels']}, "
              f"Proxy2 Target Cos: {row['training_proxy_2']['target_image_cosine']:.6f}, "
              f"Held-out Target Cos: {row['held_out']['target_image_cosine']:.6f}")

    # 2. Rankings across 189-item shared memory
    print("\n--- 2. Computing 189-item Shared Memory Rankings (20 Formal Queries) ---", flush=True)
    base_bank = validate_balanced_pool(plan["pool_path"])
    source_id = plan["source_memory_id"]
    eval_dir = PROJECT_ROOT / "output/shampoo_lucid_quickgelu_16/eval_banks"
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    encoder = MemoizedEncoder(make_eval_encoder('cuda'))
    ranking_rows = []
    try:
        for var_id, item in variants.items():
            bank = copy.deepcopy(base_bank)
            for entry in bank:
                entry['PrimaryImage']['LocalPath'] = str(resolve_image(entry, plan['pool_path']))
            if var_id != "clean":
                entry = next(e for e in bank if e['Id'] == source_id)
                entry['RetrievalImage'] = dict(
                    ASIN=plan['source_asin'],
                    LocalPath=item['image'],
                    SHA256=item['sha256']
                )
            bank_path = eval_dir / f"{var_id}.json"
            freeze_json(bank_path, bank)
            
            ranker = VisualMemoryRetriever(encoder, bank_path, cache_dir=None)
            queries_out = []
            for q in plan["queries"]:
                ranked = ranker.rank(bank, q["instruction"], str(query_img_path))
                chosen = next(r for r in ranked if r["Id"] == source_id)
                queries_out.append({
                    "task_id": q["task_id"],
                    "rank": chosen["rank"],
                    "score": chosen["score"]
                })
            
            ranks = [q["rank"] for q in queries_out]
            mean_rank = float(statistics.mean(ranks))
            top3 = sum(1 for r in ranks if r <= 3)
            top5 = sum(1 for r in ranks if r <= 5)
            print(f"[{item['name']}] Mean Rank: {mean_rank:.2f}, Top-3: {top3}/20, Top-5: {top5}/20")
            ranking_rows.append({
                "variant": var_id,
                "name": item["name"],
                "mean_rank": mean_rank,
                "top3": top3,
                "top5": top5,
                "ranks": ranks,
                "queries": queries_out
            })
    finally:
        encoder.close()
        gc.collect()

    # 3. Overall output JSON
    final_output = {
        "similarities": sim_rows,
        "rankings": ranking_rows,
        "comparison": {
            "ssim_delta": sim_rows[2]["source_ssim"] - sim_rows[1]["source_ssim"],
            "mae_delta": sim_rows[2]["source_mae_levels"] - sim_rows[1]["source_mae_levels"],
            "proxy2_target_cos_delta": sim_rows[2]["training_proxy_2"]["target_image_cosine"] - sim_rows[1]["training_proxy_2"]["target_image_cosine"],
            "heldout_target_cos_delta": sim_rows[2]["held_out"]["target_image_cosine"] - sim_rows[1]["held_out"]["target_image_cosine"],
            "mean_rank_delta": ranking_rows[2]["mean_rank"] - ranking_rows[1]["mean_rank"],
            "top3_delta": ranking_rows[2]["top3"] - ranking_rows[1]["top3"],
            "top5_delta": ranking_rows[2]["top5"] - ranking_rows[1]["top5"],
        }
    }
    
    out_file = PROJECT_ROOT / "output/shampoo_lucid_quickgelu_16/OFFICIAL_EVALUATION_REPORT.json"
    freeze_json(out_file, final_output)
    print(f"\n=== Completed successfully. Results written to {out_file} ===", flush=True)

if __name__ == "__main__":
    run()
