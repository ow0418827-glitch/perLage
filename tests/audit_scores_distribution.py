"""Investigate candidate score distributions in 189 pool under multimodal vs image-only.
"""
import json
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def analyze_distributions():
    print("=== Analyzing 189 Candidate Pool Composition and Scores ===")
    pool = json.loads((PROJECT_ROOT / "output/shared_memory/pool_189_v1/shared_memory.json").read_text(encoding="utf-8"))['entries']
    source_id = "shared_759623cdb6bb5755f91a445b0e1bf4635684824e214fa1b190e23413d6814831"
    
    # 1. Official multimodal rankings
    rankings_file = PROJECT_ROOT / "output/shampoo_ablation_20261107/evaluation/RANKINGS.json"
    rankings_data = json.loads(rankings_file.read_text(encoding="utf-8"))
    
    # Let's inspect 'clean' and 'perlage_latent_only' for task_id 'vague_01'
    for row in rankings_data['rows']:
        if row['variant'] in ['clean', 'lucid_original_16', 'perlage_latent_only']:
            vname = row['variant']
            q0 = row['queries'][0]
            top_candidates = q0['ranking'][:5]
            print(f"\n[Multimodal: {vname}] Task: {q0['task_id']} Source Rank: {q0['source_rank']} Score: {q0['source_score']:.6f}")
            print("Top 5 candidates:")
            for cand in top_candidates:
                entry = next(e for e in pool if e['Id'] == cand['Id'])
                title = entry.get('PrimaryImage', {}).get('Title', 'NO_TITLE')[:60]
                group = entry.get('SourceGroup', 'unknown')
                print(f"  Rank {cand['rank']}: ID={cand['Id'][:15]} Score={cand['score']:.6f} Group={group:<12} Title={title}")

    # 2. Image-only rankings with target_original
    img_res = json.loads((PROJECT_ROOT / "output/shampoo_ablation_20261107/image_only/RESULTS.json").read_text(encoding="utf-8"))
    g14_target_clean = next(r for r in img_res if r['model'] == 'current_g14' and r['query'] == 'target_original' and r['variant'] == 'clean')
    g14_target_latent = next(r for r in img_res if r['model'] == 'current_g14' and r['query'] == 'target_original' and r['variant'] == 'perlage_latent_only')
    
    print("\n[Image-Only target_original: clean] Source Rank:", g14_target_clean['rank'], "Score:", g14_target_clean['cosine'])
    print("Top 5 candidates in image-only (target_original, clean):")
    for cand in g14_target_clean['ranking'][:5]:
        entry = next(e for e in pool if e['Id'] == cand['Id'])
        title = entry.get('PrimaryImage', {}).get('Title', 'NO_TITLE')[:60]
        group = entry.get('SourceGroup', 'unknown')
        print(f"  Rank {cand['rank']}: ID={cand['Id'][:15]} Cosine={cand['cosine']:.6f} Group={group:<12} Title={title}")
        
    print("\n[Image-Only target_original: perlage_latent_only] Source Rank:", g14_target_latent['rank'], "Score:", g14_target_latent['cosine'])
    print("Top 5 candidates in image-only (target_original, perlage_latent_only):")
    for cand in g14_target_latent['ranking'][:5]:
        entry = next(e for e in pool if e['Id'] == cand['Id'])
        title = entry.get('PrimaryImage', {}).get('Title', 'NO_TITLE')[:60]
        group = entry.get('SourceGroup', 'unknown')
        print(f"  Rank {cand['rank']}: ID={cand['Id'][:15]} Cosine={cand['cosine']:.6f} Group={group:<12} Title={title}")

    # Who are the candidates ahead of source in target_original?
    print("\nCandidates ranked ahead of source in Image-Only (target_original, perlage_latent_only):")
    for cand in g14_target_latent['ranking'][:12]:
        entry = next(e for e in pool if e['Id'] == cand['Id'])
        title = entry.get('PrimaryImage', {}).get('Title', 'NO_TITLE')[:60]
        group = entry.get('SourceGroup', 'unknown')
        is_src = (cand['Id'] == source_id)
        print(f"  Rank {cand['rank']:>2}: {'*SRC*' if is_src else '     '} Score={cand['cosine']:.6f} Group={group:<12} Title={title}")

if __name__ == "__main__":
    analyze_distributions()
