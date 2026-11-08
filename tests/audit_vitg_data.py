"""Audit and trace data for ViT-g-14 Multimodal vs Image-Only experiments.
Strict read-only verification script placed in tests/ as required.
"""
import json
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def audit():
    print("=== 1. Checking Image-Only Results ===")
    img_only_res = json.loads((PROJECT_ROOT / "output/shampoo_ablation_20261107/image_only/RESULTS.json").read_text(encoding="utf-8"))
    print(f"Total rows in image_only RESULTS.json: {len(img_only_res)}")
    
    # Filter for current_g14
    g14_target = [r for r in img_only_res if r['model'] == 'current_g14' and r['query'] == 'target_original']
    print("\n--- current_g14 with target_original query ---")
    for r in g14_target:
        print(f"Variant: {r['variant']:<25} Label: {r['label']:<18} Rank: {r['rank']:>2}/189  Cosine: {r['cosine']:.6f}")
        
    g14_unlabeled = [r for r in img_only_res if r['model'] == 'current_g14' and r['query'] == 'unlabeled_reference']
    print("\n--- current_g14 with unlabeled_reference query ---")
    for r in g14_unlabeled:
        print(f"Variant: {r['variant']:<25} Label: {r['label']:<18} Rank: {r['rank']:>2}/189  Cosine: {r['cosine']:.6f}")

    old_target = [r for r in img_only_res if r['model'] == 'old_B32' and r['query'] == 'target_original']
    print("\n--- old_B32 with target_original query ---")
    for r in old_target:
        print(f"Variant: {r['variant']:<25} Label: {r['label']:<18} Rank: {r['rank']:>2}/189  Cosine: {r['cosine']:.6f}")

    print("\n=== 2. Checking Official Multimodal Ablation (Nov 7) ===")
    summary_path = PROJECT_ROOT / "output/shampoo_ablation_20261107/SUMMARY.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(f"Total variants in multimodal SUMMARY.json: {len(summary['rows'])}")
    for r in summary['rows']:
        print(f"Variant: {r['variant']:<32} Label: {r['label']:<20} MeanRank: {r['mean_rank']:.2f} Top1: {r['top1']}/20 Top3: {r['top3']}/20 MeanScore: {r['mean_score']:.4f}")

    print("\n=== 3. Checking Official Evaluation Report for QuickGELU (Nov 8 afternoon) ===")
    quickgelu_path = PROJECT_ROOT / "output/shampoo_lucid_quickgelu_16/OFFICIAL_EVALUATION_REPORT.json"
    if quickgelu_path.exists():
        q_data = json.loads(quickgelu_path.read_text(encoding="utf-8"))
        for r in q_data['rankings']:
            print(f"Variant: {r['variant']:<25} Name: {r['name']:<22} MeanRank: {r['mean_rank']:.2f} Top3: {r['top3']}/20 Top5: {r['top5']}/20 Ranks: {r['ranks']}")

if __name__ == "__main__":
    audit()
