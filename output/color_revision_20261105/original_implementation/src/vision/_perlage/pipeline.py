"""独立自然优先感知优化对抗方法 (PerLAGE) 的四阶段完整管线。"""
from dataclasses import asdict, dataclass, field
import gc
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image
import torch

from src.paths import PROJECT
from src.vision._natural.colour import (
    HarmonizationConfig, foreground_statistics, segment_owned_image)
from src.vision._natural.assets import NaturalGenerationConfig
from src.vision._natural.evidence import stage_record
from src.vision._natural.optimizer import InitializationOptimizationConfig
from src.vision._perlage.initialization import prepare_generation_initialization
from src.vision._perlage.latent import PerLAGELatentConfig, refine_harmonized_generation
from src.vision._perlage.evidence import METHOD, DEFINITION, validate_complete, read
from src.vision.image_utils import sha256_file, write_json
from src.vision.natural_visual_edit import EditPlan
from src.vision._perlage.harmonization import _fuse
from src.vision.retrieval_objective import TRAINING_DATA_POLICY, training_data


@dataclass(frozen=True)
class PerLAGEConfig:
    harmonization: dict = field(default_factory=lambda: dict(
        smoothing_sigma=12.0, print_ring_sigma=7.0, selected_color_mix=0.25,
        strength=0.75, blend_mode='matte'))
    latent: dict = field(default_factory=lambda: asdict(PerLAGELatentConfig()))


def validate_definition(definition):
    if definition.get('method') != METHOD or definition.get('definition') != DEFINITION:
        raise ValueError('优化方法必须使用独立的方法身份，不能作为原方法的分支')
    if any(key in definition for key in ('public_pool', 'completed_asset', 'interface', 'variant', 'study_arm')):
        raise ValueError('优化方法禁止竞争池、旧成图与分支选择')
    if definition.get('color_scheme') != 'statistics':
        raise ValueError('优化方法仍保留主体统计选色')
    training_data(definition)
    initial = definition.get('initialization', {})
    if set(initial) - {'optimizer', 'generation', 'plan'}:
        raise ValueError('初始化须在当前输出目录重新计算，不接受其他成图路径')
    optimizer = InitializationOptimizationConfig(**dict(initial.get('optimizer', {}), shadow_weight=0.0))
    if initial.get('optimizer', {}).get('shadow_weight', 0) != 0:
        raise ValueError('优化方法初始化不能启用竞争损失')
    optimizer.validate()
    defaults = PerLAGEConfig()
    config = PerLAGEConfig(dict(defaults.harmonization, **definition.get('harmonization', {})),
                       dict(defaults.latent, **definition.get('latent', {})))
    colour = HarmonizationConfig(**config.harmonization)
    colour.validate()
    if colour.blend_mode != 'matte':
        raise ValueError('保持相同的线性光颜色融合')
    PerLAGELatentConfig(**config.latent).validate()
    return config


def generate_candidate(source, target, output, *, definition, device='cuda'):
    config = validate_definition(definition)
    data = training_data(definition)
    for key, given in (('source', source), ('target', target)):
        if sha256_file(given) != sha256_file(data[key]):
            raise ValueError('配置和当前公开图对不符')
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    config_digest = hashlib.sha256(json.dumps(definition, sort_keys=True).encode('utf-8')).hexdigest()
    if (root / 'ready.json').exists():
        ready = validate_complete(read(root / 'ready.json'))
        if ready['configuration_sha256'] != config_digest:
            raise ValueError('冻结配置改变，请另建实验目录')
        return Path(ready['candidates']['final']['image']), ready
    if (root / 'PAUSE_REQUESTED').exists():
        raise InterruptedError('实验已暂停')
    tick = time.perf_counter()
    seg = root / 'foreground_segmentation'
    foreground, _ = segment_owned_image(data['source'], definition['source_boxes'], seg / 'source')
    target_fg, _ = segment_owned_image(data['target'], definition['target_boxes'], seg / 'target')
    statistics, body = foreground_statistics(data['target'], target_fg)
    stats_root = root / 'foreground_statistics'
    stats_root.mkdir(exist_ok=True)
    Image.fromarray(body.astype(np.uint8) * 255).save(stats_root / 'body_samples.png')
    write_json(stats_root / 'statistics.json', statistics)
    print(json.dumps(dict(stage='foreground_statistics', rgb=statistics['rgb'], method='PerLAGE')), flush=True)
    init = definition.get('initialization', {})
    rgb = statistics['rgb']
    plan = dict(
        instruction=(f'Give only unprinted product surfaces a subtle coherent natural finish inspired by RGB {rgb}. '
                     'Preserve the product shape, every original word, number, drawing and brand mark exactly. Preserve the scene.'),
        editable_regions=[[0, 0, 1, 1]], protected_regions=[], reason='主体统计选色后的统一生成步骤。',
        region_policy='label_protected', feather_pixels=3.0)
    initial_root = root / 'initialization'
    _, initial = prepare_generation_initialization(
        **data, output=initial_root,
        plan=EditPlan(**init.get('plan', plan)), surface_regions=definition['print_regions'],
        config=InitializationOptimizationConfig(**dict(init.get('optimizer', {}), shadow_weight=0.0)),
        generation=NaturalGenerationConfig(**init.get('generation', {})), device=device,
        pause_path=root / 'PAUSE_REQUESTED')
    from src.vision.dual_rag_bridge import make_encoder
    from src.vision.diffusion_purification import LatentDiffusionPurifier, PurificationConfig
    base = make_encoder(device=device)
    fused, _ = _fuse(data, initial, foreground, statistics, definition['print_regions'], root, config, base)
    purifier = LatentDiffusionPurifier(PurificationConfig(working_size=512), device=device)
    purifier.vae.enable_gradient_checkpointing()
    print('初始化与统计颜色融合完成，开始生成感知优化潜变量细化。', flush=True)
    image, meta = refine_harmonized_generation(
        **data, anchor=fused / 'generative_anchor.png',
        initial=fused / 'initial.png', teacher=initial['candidates']['natural_final']['image'],
        mask_path=fused / 'edit_mask.png', source_foreground=seg / 'source/foreground.png',
        parent_anchor=initial['candidates']['generation_only']['image'], print_regions=definition['print_regions'],
        output=root / 'latent_generation_refinement', surrogate=base.surrogate, purifier=purifier,
        config=PerLAGELatentConfig(**config.latent), pause_path=root / 'PAUSE_REQUESTED')
    if meta['status'] != 'completed':
        raise InterruptedError('优化方法已保存潜变量细化断点')
    final, anchor = root / 'adversarial.png', root / 'generative_anchor.png'
    shutil.copyfile(image, final)
    shutil.copyfile(root / 'latent_generation_refinement/generative_anchor.png', anchor)
    init_identity = initial_root / 'run_identity.json'
    optimizer_identity = initial_root / 'attack/identity.json'
    ready = dict(
        method=METHOD, definition=DEFINITION, status='completed', color_scheme='statistics',
        training_data_policy=TRAINING_DATA_POLICY, shadow_pool_used=False, purchase_history_used=False,
        victim_memory_used=False, formal_queries_used=False, configuration_sha256=config_digest,
        source_sha256=sha256_file(source), target_sha256=sha256_file(target),
        source_title_sha256=hashlib.sha256(data['source_title'].encode('utf-8')).hexdigest(),
        candidates={'final': dict(image=str(final), sha256=sha256_file(final))},
        initialization_identity=str(init_identity), initialization_optimizer_identity=str(optimizer_identity),
        initialization_evidence={str(p): sha256_file(p) for p in (init_identity, optimizer_identity, initial_root / 'ready.json')},
        stages=dict(
            foreground_segmentation=stage_record([
                seg / 'source/record.json', seg / 'source/foreground.png',
                seg / 'target/record.json', seg / 'target/foreground.png']),
            foreground_statistics=stage_record([
                stats_root / 'statistics.json', stats_root / 'body_samples.png']),
            color_harmonization=stage_record([
                fused / 'metadata.json', fused / 'generative_anchor.png',
                fused / 'initial.png', fused / 'edit_mask.png']),
            latent_generation_refinement=stage_record([
                root / 'latent_generation_refinement/metadata.json',
                root / 'latent_generation_refinement/identity.json', final, anchor],
                steps_completed=max(row['step'] for row in meta['history']), steps_requested=config.latent['steps'])
        ),
        observed_seconds=time.perf_counter() - tick,
        auxiliary_max_levels=meta['auxiliary_max_levels'],
        latent_config=config.latent,
        source_method='four_natural_stages_with_perceptual_latent_optimization'
    )
    validate_complete(ready, final)
    write_json(root / 'ready.json', ready)
    del base, purifier
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return final, ready
