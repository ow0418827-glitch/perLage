"""针对第二组（02_chair）的 PerLAGE_no_pixel（无像素扰动感知生成编辑方法）独立成图与凭证审计脚本。"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

# 设置项目根目录与离线运行环境变量
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

os.environ.update(
    HF_HUB_OFFLINE='1',
    TRANSFORMERS_OFFLINE='1',
    TOKENIZERS_PARALLELISM='false',
    PYTHONUTF8='1'
)

# 确保在 Windows 控制台输出安全（避免 UnicodeEncodeError）
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import numpy as np
from src.paths import PROJECT, OUTPUT, ProjectPath
from src.visual.optimization import generate_attack
from src.visual.catalog_training_guard import catalog_training_guard
from src.visual.audit import digest, freeze_json, read_json, save_json
from src.visual.image_utils import sha256_file
from src.visual.attack_completion import validate_attack_completion
from src.visual._perlage_no_pixel.evidence import validate_complete, METHOD, DEFINITION
from src.visual.retrieval_objective import TRAINING_DATA_POLICY
from src.visual._perlage.components.colour import read_rgb
from src.experiments.pipeline import source_snapshot


def main():
    print(f"=== [02_chair] 开始执行 {METHOD} 无像素扰动感知生成任务 ===", flush=True)

    # 1. 路径定义与输入资产校验
    output_root = OUTPUT / 'chair_ablation_20261108'
    generation_dir = output_root / 'generation'
    public_chair_dir = OUTPUT / 'datasets/public/groups/02_chair'

    source_img = public_chair_dir / 'source.jpg'
    target_img = public_chair_dir / 'target.jpg'

    EXPECTED_SOURCE_SHA256 = '47a70fad01c018bb524eefcf803794826e4a48e1b667cd77ba0ccecc1c28ecfa'
    EXPECTED_TARGET_SHA256 = '8fc2834a77b8ecf235f5bec9c1f335830d1de142843cda13284cdfe36579000d'

    if not source_img.exists() or not target_img.exists():
        raise FileNotFoundError(f"公开图对输入不存在: source={source_img}, target={target_img}")

    src_hash = sha256_file(source_img)
    tgt_hash = sha256_file(target_img)
    assert src_hash == EXPECTED_SOURCE_SHA256, f"源图片 SHA256 异常: {src_hash} != {EXPECTED_SOURCE_SHA256}"
    assert tgt_hash == EXPECTED_TARGET_SHA256, f"目标图片 SHA256 异常: {tgt_hash} != {EXPECTED_TARGET_SHA256}"
    print(f"公开图对核验完全通过: source={src_hash[:10]}..., target={tgt_hash[:10]}...", flush=True)

    output_root.mkdir(parents=True, exist_ok=True)
    generation_dir.mkdir(parents=True, exist_ok=True)

    # 2. 组装符合严格零像素规范的 definition 字典
    source_title = (
        "FMOGG Hydraulic Salon Recliner Barber Chair Heavy Duty Hair Spa Salon Styling Beauty, "
        "360°Rotating, Explosion-Proof Pneumatic Rod Tattoo Chair for Salon Beauty Equipment (Color : A)"
    )
    training_queries = [
        "Find a product with a similar overall visual appearance.",
        "Use the reference picture as a broad visual guide for product matching.",
        "Look for an item with a comparable general shape and finish."
    ]

    definition = {
        'method': METHOD,
        'definition': DEFINITION,
        'color_scheme': 'statistics',
        'training_data_policy': TRAINING_DATA_POLICY,
        'pixel_optimization': False,
        'data': {
            'source': str(source_img.resolve()),
            'target': str(target_img.resolve()),
            'source_title': source_title,
            'training_queries': training_queries,
        },
        'source_boxes': [[75, 15, 458, 485]],
        'target_boxes': [[3, 4, 399, 496]],
        'print_regions': [],
        'initialization': {
            'optimizer': {
                'shadow_weight': 0.0,
                'pixel_clean_steps': 0,
                'pixel_diffusion_steps': 0,
                'pixel_levels': 0,
                'latent_clean_steps': 40,
                'latent_diffusion_steps': 40,
                'check_every': 20,
            },
            'generation': {
                'seed': 42,
                'target_scale': 8.0,
                'latent_clip': 0.04,
                'strength': 1.0,
                'steps': 30,
                'image_size': 384,
                'source_weight': 1.0,
                'robust_guidance': True,
            }
        },
        'harmonization': {
            'smoothing_sigma': 12.0,
            'print_ring_sigma': 7.0,
            'selected_color_mix': 0.25,
            'strength': 0.75,
            'blend_mode': 'matte',
        },
        'latent': {
            'steps': 60,
            'check_every': 10,
            'latent_levels': 24,
            'latent_working_size': 256,
            'latent_limit': 0.5,
            'latent_learning_rate': 0.01,
            'latent_smoothing_mix': 0.12,
            'auxiliary_levels': 0,
            'source_ssim_floor': 0.85,
            'source_mae_ceiling': 12.0,
            'ordinary_tolerance': 0.01,
            'temperature': 0.01,
            'desired_owned_score_gain': 0.02,
            'feature_weight': 0.02,
            'diffusion_weight': 3.0,
            'macro_l2_weight': 0.001,
            'macro_tv_weight': 0.003,
            'guard_backtracks': 8,
            'full_eot_samples': 2,
            'training_seed_base': 720300000,
            'timesteps': [99, 199, 399],
            'development_seeds': [9401, 9402, 9403, 9404, 9405, 9406, 9407, 9408],
            'selection_seeds': [9301, 9302],
            'retention_tolerance': 0.0005,
        }
    }

    # 3. 运行受沙箱保护的生成流水线（若已有 ready.json 则直接加载验证）
    start_time = time.perf_counter()
    with catalog_training_guard(allowed_output_roots=[output_root, OUTPUT / 'datasets']) as audit:
        image, metadata = generate_attack(
            source=source_img,
            target=target_img,
            output_dir=generation_dir,
            method=METHOD,
            perlage_no_pixel_config=definition,
            device='cuda',
        )
        freeze_json(generation_dir / 'completion.json', metadata)

    audit_file = generation_dir / 'TRAINING_ACCESS_AUDIT.json'
    if audit_file.exists():
        audit_clean = read_json(audit_file)
    else:
        audit_clean = {k: v for k, v in audit.items() if k != 'active'}
        freeze_json(audit_file, audit_clean)
    freeze_json(output_root / 'TRAINING_ACCESS_AUDIT.json', audit_clean)
    elapsed = time.perf_counter() - start_time
    print(f"=== 四阶段生成执行完成，耗时 {elapsed:.2f} 秒 ===", flush=True)

    # 4. 严格断言检查
    print("=== 开始执行零像素与零残差断言检查 ===", flush=True)
    adversarial_png = generation_dir / 'adversarial.png'
    generative_anchor_png = generation_dir / 'generative_anchor.png'

    final_rgb = read_rgb(adversarial_png)
    anchor_rgb = read_rgb(generative_anchor_png)
    assert np.array_equal(final_rgb, anchor_rgb), (
        "致命错误：adversarial.png 与 generative_anchor.png 数组不相等，存在像素级加性扰动！"
    )
    max_diff = np.abs(final_rgb.astype(np.int32) - anchor_rgb.astype(np.int32)).max()
    assert max_diff == 0, f"像素绝对残差最大值非零: {max_diff}"
    print(f"[PASS] 逐像素全等核验通过: np.array_equal == True, max_diff == {max_diff}")

    # 5. 图片冻结与根目录对齐
    frozen_png = output_root / 'frozen.png'
    shutil.copyfile(adversarial_png, frozen_png)
    shutil.copyfile(adversarial_png, output_root / 'adversarial.png')
    shutil.copyfile(generative_anchor_png, output_root / 'generative_anchor.png')
    shutil.copyfile(generation_dir / 'ready.json', output_root / 'ready.json')
    shutil.copyfile(generation_dir / 'completion.json', output_root / 'completion.json')
    freeze_json(output_root / 'definition.json', definition)

    adv_hash = sha256_file(adversarial_png)
    frozen_hash = sha256_file(frozen_png)
    anchor_hash = sha256_file(generative_anchor_png)
    assert adv_hash == frozen_hash, f"对抗图片与冻结图片哈希不一致: {adv_hash} != {frozen_hash}"
    assert adv_hash == anchor_hash, f"对抗图片与生成锚点哈希不一致: {adv_hash} != {anchor_hash}"
    print(f"[PASS] SHA-256 哈希完全一致: {adv_hash}")

    # 6. 构建 PUBLIC_RUN.json 与 FROZEN_IMAGE.json (13 个顶级键)
    pair_manifest_path = public_chair_dir / 'pair.json'
    pair_data = read_json(pair_manifest_path)

    # 组装精确的 source 与 target 描述结构
    source_spec = copy.deepcopy(pair_data['source'])
    source_spec['image'] = str(source_img.resolve())
    source_spec['foreground_mask'] = str((public_chair_dir / 'metadata/source_foreground.png').resolve())

    target_spec = copy.deepcopy(pair_data['target'])
    target_spec['image'] = str(target_img.resolve())
    target_spec['foreground_mask'] = str((public_chair_dir / 'metadata/target_foreground.png').resolve())

    public_binding = {
        'method': METHOD,
        'selection_sha256': sha256_file(pair_manifest_path),
        'source': source_spec,
        'target': target_spec,
        'options': {},
        'source_files': source_snapshot(),
        'shared_pool_scope': 'shared',
        'threat_model': 'black_box_provisional',
    }
    freeze_json(output_root / 'PUBLIC_RUN.json', public_binding)
    binding_hash = digest(public_binding)

    frozen_image_data = {
        'schema_version': 1,
        'method': METHOD,
        'status': 'frozen',
        'image': str(frozen_png.resolve()),
        'image_sha256': frozen_hash,
        'source': source_spec,
        'target': target_spec,
        'metadata': metadata,
        'requested_steps': 60,
        'public_binding_sha256': binding_hash,
        'shared_pool_scope': 'shared',
        'victim_memory_used': False,
        'formal_queries_used': False,
    }

    assert len(frozen_image_data) == 13, f"FROZEN_IMAGE.json 键数量异常: {len(frozen_image_data)} != 13"
    freeze_json(output_root / 'FROZEN_IMAGE.json', frozen_image_data)
    freeze_json(generation_dir / 'FROZEN_IMAGE.json', frozen_image_data)
    print("[PASS] FROZEN_IMAGE.json 生成并成功冻结 (13 个顶级键已全部就绪)")

    # 7. 严格凭证审计与统一完成验证
    print("=== 开始执行独立凭据审计与统一完成验证 ===", flush=True)
    ready = read_json(generation_dir / 'ready.json')
    validate_complete(ready, frozen_png, method=METHOD)
    print("[PASS] validate_complete 审计通过！")

    validate_attack_completion(
        metadata,
        requested_steps=60,
        expected={
            'source_image_sha256': src_hash,
            'target_image_sha256': tgt_hash,
            'method': METHOD,
        }
    )
    print("[PASS] validate_attack_completion 统一完成验证通过！")

    # 8. 产出最终执行摘要
    summary = {
        'method': METHOD,
        'group': '02_chair',
        'definition': DEFINITION,
        'source_sha256': src_hash,
        'target_sha256': tgt_hash,
        'adversarial_png_sha256': adv_hash,
        'frozen_png_sha256': frozen_hash,
        'generative_anchor_png_sha256': anchor_hash,
        'max_pixel_residual': int(max_diff),
        'pixel_equal': bool(np.array_equal(final_rgb, anchor_rgb)),
        'elapsed_seconds': ready.get('observed_seconds', elapsed),
        'frozen_keys_count': len(frozen_image_data),
        'public_binding_sha256': binding_hash,
        'checked_accesses': audit_clean['checked_accesses'],
        'blocked_accesses': audit_clean['blocked_accesses'],
        'status': 'completed',
    }
    freeze_json(output_root / 'GENERATION_SUMMARY.json', summary)
    print("=== 生成摘要 ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("=== [02_chair] 任务全部圆满完成 ===", flush=True)


if __name__ == '__main__':
    main()
