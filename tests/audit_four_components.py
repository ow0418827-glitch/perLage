"""Break down the 4 cross-modal and intramodal components for ViT-g-14.
"""
import json
import numpy as np
from pathlib import Path
import torch

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def analyze_components():
    from src.experiments.shared_evaluation import make_eval_encoder
    from src.visual.visual_retrieval import resolve_image, unit
    from src.agent.memory.balanced_pool import validate_balanced_pool
    from src.experiments.formal_queries import VAGUE_INSTRUCTIONS
    
    encoder = make_eval_encoder('cuda')
    pool = validate_balanced_pool(PROJECT_ROOT / "output/shared_memory/pool_189_v1/shared_memory.json")
    query_img_path = PROJECT_ROOT / "output/shampoo_ablation_20261107/evaluation_inputs/query.png"
    target_img_path = PROJECT_ROOT / "output/shampoo_ablation_20261107/data/07_shampoo/target.jpg"
    source_img_path = PROJECT_ROOT / "output/shampoo_ablation_20261107/data/07_shampoo/source.jpg"
    latent_img_path = PROJECT_ROOT / "output/shampoo_ablation_20261107/generation/perlage_latent_only/adversarial.png"
    lucid_img_path = PROJECT_ROOT / "output/shampoo_ablation_20261107/images/lucid_original_16.png"

    source_id = "shared_759623cdb6bb5755f91a445b0e1bf4635684824e214fa1b190e23413d6814831"
    src_entry = next(e for e in pool if e['Id'] == source_id)
    src_title = src_entry['PrimaryImage']['Title']

    # Encode query elements
    q_inst = VAGUE_INSTRUCTIONS[0] # "请购买与图片中商品相似的商品。"
    fv_q = encoder.encode(str(query_img_path))
    ft_q = encoder.encode_text(q_inst)

    # Encode target elements
    fv_target = encoder.encode(str(target_img_path))

    # Evaluate variants
    variants = [
        ('clean', source_img_path),
        ('lucid_16', lucid_img_path),
        ('perlage_latent', latent_img_path)
    ]

    print(f"=== Query: '{q_inst}' ===")
    print(f"=== Source Title: '{src_title}' ===")

    for vname, vpath in variants:
        fv_k = encoder.encode(str(vpath))
        ft_k = encoder.encode_text(src_title)

        # 4 terms
        ii = float(fv_k @ fv_q)
        it = float(fv_k @ ft_q)
        ti = float(ft_k @ fv_q)
        tt = float(ft_k @ ft_q)

        # denominator norms
        norm_q = float(np.linalg.norm(fv_q + ft_q))
        norm_k = float(np.linalg.norm(fv_k + ft_k))

        joint_score = (ii + it + ti + tt) / (norm_q * norm_k)
        
        # Also image cosine to target
        cos_tgt = float(fv_k @ fv_target)

        print(f"\n[{vname}]")
        print(f"  Image-Image (fv_k @ fv_q, query.png): {ii:.6f}")
        print(f"  Image-Text  (fv_k @ ft_q, instruction): {it:.6f}")
        print(f"  Text-Image  (ft_k @ fv_q, query.png): {ti:.6f}")
        print(f"  Text-Text   (ft_k @ ft_q, instruction): {tt:.6f}")
        print(f"  Numerator sum: {ii + it + ti + tt:.6f}")
        print(f"  norm_k: {norm_k:.4f}, norm_q: {norm_q:.4f}")
        print(f"  Fused Joint Score: {joint_score:.6f}")
        print(f"  Image-Target Cosine (fv_k @ fv_target): {cos_tgt:.6f}")

    # Check top competitors from pool
    print("\n=== Checking Competitor Candidates ===")
    comp_ids = [
        ("Rank2 Jeans", "public_99e83410"),
        ("Rank5 Truss Shampoo", "shared_2002d56e"),
        ("Top Img GIOVANNI", "shared_7cb3db93")
    ]
    for label, cid_prefix in comp_ids:
        entry = next(e for e in pool if e['Id'].startswith(cid_prefix))
        c_title = entry['PrimaryImage']['Title']
        c_path = resolve_image(entry, PROJECT_ROOT / "output/shared_memory/pool_189_v1/shared_memory.json")
        fv_c = encoder.encode(str(c_path))
        ft_c = encoder.encode_text(c_title)
        
        ii = float(fv_c @ fv_q)
        it = float(fv_c @ ft_q)
        ti = float(ft_c @ fv_q)
        tt = float(ft_c @ ft_q)
        norm_c = float(np.linalg.norm(fv_c + ft_c))
        norm_q = float(np.linalg.norm(fv_q + ft_q))
        joint_score = (ii + it + ti + tt) / (norm_q * norm_c)
        cos_tgt = float(fv_c @ fv_target)

        print(f"\n[{label}: {c_title[:45]}]")
        print(f"  Image-Image (query.png): {ii:.6f}")
        print(f"  Image-Text (instruction): {it:.6f}")
        print(f"  Text-Image (query.png): {ti:.6f}")
        print(f"  Text-Text (instruction): {tt:.6f}")
        print(f"  Fused Joint Score: {joint_score:.6f}")
        print(f"  Image-Target Cosine (target.jpg): {cos_tgt:.6f}")

if __name__ == "__main__":
    analyze_components()
