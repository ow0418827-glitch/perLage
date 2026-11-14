"""从公开原图执行完整方法；初始化和最终细化分进程释放显存。"""
from __future__ import annotations
import os
for name in ['OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS']:
    os.environ[name] = '4'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(ROOT / 'implementation'))
import numpy as np
from PIL import Image
import torch
from src.vision._perlage import pipeline, initialization
from src.vision._perlage.evidence import validate_complete
from src.vision._natural.assets import NaturalGenerationConfig
from src.vision._natural.optimizer import InitializationOptimizationConfig
from src.vision._natural.mask import SurfaceMaskConfig
from src.vision.natural_visual_edit import EditPlan
from src.vision.catalog_training_guard import catalog_training_guard
from src.vision.image_utils import sha256_file
import trial_components
import latent_full


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + '.tmp')
    pending.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    pending.replace(path)


def verify_source_version(plan):
    for name, expected in plan['active_vision_source_sha256'].items():
        if sha256_file(PROJECT / name) != expected:
            raise ValueError('原方法代码在试验期间改变：' + name)
    frozen = read(ROOT / 'IMPLEMENTATION_FROZEN.json')
    for name, expected in frozen['files'].items():
        if sha256_file(ROOT / name) != expected:
            raise ValueError('独立试验代码身份改变：' + name)


def prepare_run(arm, group, plan):
    case = next(c for c in plan['cases'] if c['key'] == group)
    definition = read(case['definition'])
    run = ROOT / 'runs' / arm / group / 'generation'
    run.mkdir(parents=True, exist_ok=True)
    data = definition['data']
    for side in ['source', 'target']:
        if sha256_file(data[side]) != case['inputs_sha256'][side]:
            raise ValueError('公开原图改变')
    foreground, _ = pipeline.segment_owned_image(data['source'], definition['source_boxes'], run / 'foreground_segmentation/source')
    target_fg, _ = pipeline.segment_owned_image(data['target'], definition['target_boxes'], run / 'foreground_segmentation/target')
    if arm == 'dominant':
        statistics, body = trial_components.dominant_statistics(data['target'], target_fg)
        color = statistics['rgb']
        definition['initialization'] = {'plan': dict(
            instruction=(f'Match the unprinted material surfaces to the target product dominant color RGB {color}. '
                         'Preserve the product shape, every original word, number, drawing and brand mark exactly. Preserve the scene.'),
            editable_regions=[[0,0,1,1]], protected_regions=[],
            reason='目标前景主色引导完整生成；保护实际印刷核心并释放可改底材。',
            region_policy='label_protected', feather_pixels=3.)}
        trial_components.RUN = run
        initialization.build_surface_masks = trial_components.reviewed_surface_masks
        pipeline.foreground_statistics = trial_components.dominant_statistics
        pipeline._fuse = trial_components.dominant_fuse
        def finish_refinement(**kwargs):
            before_color = kwargs['teacher']
            kwargs['teacher'] = kwargs['initial']
            return latent_full.refine_harmonized_generation(**kwargs, halo_reference=before_color)
        pipeline.refine_harmonized_generation = finish_refinement
        trial_components.reviewed_surface_masks(data['source'], definition['print_regions'], run / 'initialization/content_masks', SurfaceMaskConfig())
    else:
        statistics, body = pipeline.foreground_statistics(data['target'], target_fg)
    stats = run / 'foreground_statistics'
    stats.mkdir(exist_ok=True)
    Image.fromarray(body.astype(np.uint8)*255).save(stats / 'body_samples.png')
    save(stats / 'statistics.json', statistics)
    config_path = run.parent / 'definition.json'
    if config_path.exists() and read(config_path) != definition:
        raise ValueError('本次完整生成配置改变')
    save(config_path, definition)
    save(run / 'PREPARED.json', dict(status='prepared', arm=arm, group=group,
         definition_sha256=sha256_file(config_path), selected_rgb=statistics['rgb'],
         segmentation_recomputed_in_this_run=True, prior_generated_images_used=False))
    return run, definition, statistics


def complete_initialization(run, definition, statistics):
    data = definition['data']
    init = definition.get('initialization', {})
    color = statistics['rgb']
    default_plan = dict(
        instruction=(f'Give only unprinted product surfaces a subtle coherent natural finish inspired by RGB {color}. '
                     'Preserve the product shape, every original word, number, drawing and brand mark exactly. Preserve the scene.'),
        editable_regions=[[0,0,1,1]], protected_regions=[], reason='主体统计选色后的统一生成步骤。',
        region_policy='label_protected', feather_pixels=3.)
    _, ready = initialization.prepare_generation_initialization(
        **data, output=run / 'initialization', plan=EditPlan(**init.get('plan', default_plan)),
        surface_regions=definition['print_regions'],
        config=InitializationOptimizationConfig(**dict(init.get('optimizer', {}), shadow_weight=0.)),
        generation=NaturalGenerationConfig(**init.get('generation', {})), device='cuda',
        pause_path=run / 'PAUSE_REQUESTED')
    evidence = read(run / 'initialization/attack/metadata.json')
    expected = dict(latent_clean=40, latent_diffusion=40, pixel_clean=20, pixel_diffusion=60)
    for phase, count in expected.items():
        completed = max(r['step'] for r in evidence['history'] if r['phase'] == phase)
        if completed != count:
            raise ValueError('初始化日程没有完整执行：' + phase)
    generation = read(run / 'initialization/generation/metadata.json')
    if generation['completed_steps'] != 30:
        raise ValueError('指令生成三十步未完成')
    save(run / 'INITIALIZATION_COMPLETED.json', dict(status='completed',
         instruction_generation_steps=generation['completed_steps'], optimization_steps=expected,
         ready_sha256=sha256_file(run / 'initialization/ready.json'),
         initialized_from_this_runs_original_public_images=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['original', 'dominant'], required=True)
    parser.add_argument('--group', choices=['07_shampoo', '09_toothbrush'], required=True)
    parser.add_argument('--phase', choices=['prepare', 'initialize', 'finish'], required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    plan = read(ROOT / 'PLAN.json')
    verify_source_version(plan)
    run = ROOT / 'runs' / args.arm / args.group / 'generation'
    started = time.perf_counter()
    error = None
    state = {}
    try:
        with catalog_training_guard(allowed_output_roots=[ROOT], private_paths=[ROOT / 'controls', PROJECT.parent / 'lage', PROJECT.parent / 'MINJA-main']) as state:
            run, definition, statistics = prepare_run(args.arm, args.group, plan)
            if args.phase == 'initialize':
                complete_initialization(run, definition, statistics)
            elif args.phase == 'finish':
                if not (run / 'INITIALIZATION_COMPLETED.json').is_file():
                    raise ValueError('禁止跳过本次完整初始化')
                image, ready = pipeline.generate_candidate(definition['data']['source'], definition['data']['target'],
                    run, definition=definition, device='cuda')
                validate_complete(ready, image)
                latent = read(run / 'latent_generation_refinement/metadata.json')
                if max(r['step'] for r in latent['history']) != 60:
                    raise ValueError('最终六十步潜变量细化未完成')
                if not latent['best']['quality_ok'] or not latent['best']['natural_halo_gate']:
                    raise ValueError('完整图未通过原有质量与印刷光晕检查')
                raw = np.asarray(Image.open(image).convert('RGB')).astype(int)
                source = np.asarray(Image.open(definition['data']['source']).convert('RGB')).astype(int)
                core = np.asarray(Image.open(run / 'initialization/content_masks/core.png').convert('L')) > 0
                delta = int(np.abs(raw-source)[core].max()) if core.any() else 0
                if delta != 0:
                    raise ValueError('完整成图改变受保护的源印刷核心')
                save(run.parent / 'FULL_METHOD_COMPLETED.json', dict(status='completed', arm=args.arm,
                    group=args.group, experimental_protocol=plan['protocol'], image=str(image),
                    image_sha256=sha256_file(image), ready_sha256=sha256_file(run / 'ready.json'),
                    all_four_stages_completed=True, final_latent_steps=60,
                    initialization_steps=[40,40,20,60], instruction_generation_steps=30,
                    protected_core_max_difference=delta, reused_old_initialization=False,
                    quality=latent['best'], registered_as_main_method=False))
    except Exception as exc:
        error = dict(type=type(exc).__name__, message=str(exc))
        raise
    finally:
        save(run.parent / ('TRAINING_ACCESS_' + args.phase + '.json'), dict(
            status='completed' if error is None else 'failed', phase=args.phase,
            elapsed_seconds=time.perf_counter()-started,
            audit=state, error=error, public_images_and_names_only=True,
            prior_color_step_decision_used_to_stop_full_generation=False))


if __name__ == '__main__':
    main()
