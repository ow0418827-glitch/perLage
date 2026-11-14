"""独立无像素扰动感知优化对抗方法 (PerLAGE_no_pixel) 的四阶段完整管线。"""
from dataclasses import asdict, dataclass, field
import gc
import hashlib
import json
from src.paths import ProjectPath as Path
import shutil
import time

import numpy as np
from PIL import Image
import torch

from src.paths import PROJECT
from src.visual._perlage.components.colour import (
    HarmonizationConfig, foreground_statistics, segment_owned_image, read_rgb)
from src.visual._perlage.components.assets import NaturalGenerationConfig
from src.visual._perlage.components.evidence import stage_record
from src.visual._perlage.components.optimizer import InitializationOptimizationConfig
from src.visual._perlage.initialization import prepare_generation_initialization
from src.visual._perlage.latent import PerLAGELatentConfig, refine_harmonized_generation
from src.visual._perlage_no_pixel.evidence import (
    METHOD, DEFINITION, validate_complete, read)
from src.visual.image_utils import read_tensor, sha256_file, write_json
from src.visual._perlage.components.instruction import EditPlan
from src.visual._perlage.harmonization import _fuse
from src.visual.retrieval_objective import TRAINING_DATA_POLICY, training_data


@dataclass(frozen=True)
class PerLAGENoPixelConfig:
    harmonization: dict = field(default_factory=lambda: dict(
        smoothing_sigma=12.0, print_ring_sigma=7.0, selected_color_mix=0.25,
        strength=1.0, blend_mode='matte'))
    latent: dict = field(default_factory=lambda: dict(asdict(PerLAGELatentConfig()), auxiliary_levels=0))
    pixel_optimization: bool = False


def validate_definition(definition):
    method = definition.get('method', METHOD)
    if method not in (METHOD, 'perlage_no_pixel'):
        raise ValueError(f'无像素方法必须使用独立的方法身份 {METHOD}')
    if definition.get('definition', DEFINITION) != DEFINITION:
        raise ValueError(f'无像素方法定义必须为 {DEFINITION}')
    if any(key in definition for key in ('public_pool', 'completed_asset', 'interface', 'variant', 'study_arm')):
        raise ValueError('优化方法禁止竞争池、旧成图与分支选择')
    if definition.get('color_scheme', 'statistics') != 'statistics':
        raise ValueError('优化方法仍保留主体统计选色')
    training_data(definition)
    initial = definition.get('initialization', {})
    if set(initial) - {'optimizer', 'generation', 'plan'}:
        raise ValueError('初始化须在当前输出目录重新计算，不接受其他成图路径')
    if initial.get('optimizer', {}).get('shadow_weight', 0) != 0:
        raise ValueError('优化方法初始化不能启用竞争损失')

    # Strict zero-pixel / zero-residual assertions on definition
    if definition.get('pixel_optimization', False) is not False:
        raise ValueError('无像素方法禁止开启像素优化')
    if definition.get('latent', {}).get('auxiliary_levels', 0) != 0:
        raise ValueError('仅潜变量模式禁止像素优化：固定像素残差')

    opt = initial.get('optimizer', {})
    for key in ('pixel_clean_steps', 'pixel_diffusion_steps', 'pixel_levels'):
        if opt.get(key, 0) != 0:
            raise ValueError(f'无像素模式不能配置像素优化：{key}')

    optimizer = initialization_config(definition)
    optimizer.validate()
    assert optimizer.pixel_clean_steps == 0, 'pixel_clean_steps 必须为 0'
    assert optimizer.pixel_diffusion_steps == 0, 'pixel_diffusion_steps 必须为 0'
    assert optimizer.pixel_levels == 0, 'pixel_levels 必须为 0'

    defaults = PerLAGENoPixelConfig()
    latent = dict(defaults.latent, **definition.get('latent', {}))
    latent['auxiliary_levels'] = 0
    config = PerLAGENoPixelConfig(
        dict(defaults.harmonization, **definition.get('harmonization', {})),
        latent,
        pixel_optimization=False
    )
    colour = HarmonizationConfig(**config.harmonization)
    colour.validate()
    if colour.blend_mode != 'matte':
        raise ValueError('保持相同的线性光颜色融合')
    PerLAGELatentConfig(**config.latent).validate()
    assert config.pixel_optimization is False, '无像素方法 pixel_optimization 必须为 False'
    assert config.latent['auxiliary_levels'] == 0, '无像素方法 auxiliary_levels 必须为 0'
    return config


def initialization_config(definition):
    """无像素感知方法强制将所有像素优化参数与预算锁死为 0。"""
    method = definition.get('method', METHOD)
    if method not in (METHOD, 'perlage_no_pixel'):
        raise ValueError('初始化配置与独立方法身份不一致')
    values = dict(definition.get('initialization', {}).get('optimizer', {}), shadow_weight=0.0)
    for key in ('pixel_clean_steps', 'pixel_diffusion_steps', 'pixel_levels'):
        if values.get(key, 0) != 0:
            raise ValueError('仅潜变量模式不能配置像素优化：' + key)
        values[key] = 0
    config = InitializationOptimizationConfig(**values)
    config.validate()
    assert config.pixel_clean_steps == 0, 'pixel_clean_steps 必须为 0'
    assert config.pixel_diffusion_steps == 0, 'pixel_diffusion_steps 必须为 0'
    assert config.pixel_levels == 0, 'pixel_levels 必须为 0'
    return config


def generate_perlage_no_pixel(source, target, output, *, definition, device='cuda'):
    """独立无像素扰动感知方法专用四阶段执行流水线。
    强制断言初始化与细化各阶段像素步数和残差为 0。
    """
    definition = dict(definition)
    if definition.get('method') in (None, 'perlage_no_pixel'):
        definition['method'] = METHOD
    if definition.get('definition') is None:
        definition['definition'] = DEFINITION
    definition['pixel_optimization'] = False
    definition.setdefault('latent', {})['auxiliary_levels'] = 0

    config = validate_definition(definition)
    assert config.pixel_optimization is False, '无像素方法配置必须为 pixel_optimization=False'
    assert config.latent['auxiliary_levels'] == 0, '无像素方法配置必须为 auxiliary_levels=0'

    data = training_data(definition)
    for key, given in (('source', source), ('target', target)):
        if sha256_file(given) != sha256_file(data[key]):
            raise ValueError('配置和当前公开图对不符')

    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    config_digest = hashlib.sha256(json.dumps(definition, sort_keys=True).encode('utf-8')).hexdigest()
    identity_path = root / 'method_identity.json'
    implementations = [
        'src/visual/_perlage_no_pixel/pipeline.py',
        'src/visual/_perlage_no_pixel/evidence.py',
        'src/visual/_perlage/initialization.py',
        'src/visual/_perlage/latent.py',
        'src/visual/_perlage/components/optimizer.py',
        'src/visual/_perlage/components/core.py',
        'src/visual/_perlage/components/assets.py',
        'src/visual/_perlage/components/editor.py',
        'src/visual/_perlage/components/instruction.py',
        'src/visual/_perlage/components/mask.py',
        'src/visual/_perlage/components/colour.py',
        'src/visual/_perlage/components/segmentation.py',
        'src/visual/_perlage/components/losses.py',
        'src/visual/_perlage/components/latent_branch.py',
        'src/visual/_perlage/harmonization.py',
        'src/visual/encoders/backbone.py',
        'src/visual/encoders/clip.py',
        'src/visual/encoders/joint.py',
        'src/visual/retrieval_objective.py',
        'src/visual/robust_objectives.py',
        'src/visual/diffusion_components.py',
        'src/visual/diffusion_purification.py',
        'src/visual/config/optimization.json',
        'src/model_identity.py'
    ]
    identity = dict(
        method=METHOD,
        definition=DEFINITION,
        configuration_sha256=config_digest,
        source_sha256=sha256_file(source),
        target_sha256=sha256_file(target),
        implementation={name: sha256_file(PROJECT / name) for name in implementations if (PROJECT / name).is_file()}
    )
    if identity_path.exists():
        if read(identity_path) != identity:
            raise ValueError('独立方法、输入或实现已改变，请使用新的输出目录')
    elif any(root.iterdir()):
        raise ValueError('非空目录没有本次独立方法身份，禁止复用旧图或旧断点')
    else:
        write_json(identity_path, identity)

    if (root / 'ready.json').exists():
        ready = validate_complete(read(root / 'ready.json'), method=METHOD)
        if ready['configuration_sha256'] != config_digest:
            raise ValueError('冻结配置改变，请另建实验目录')
        if ready.get('pixel_optimization', True) is not False:
            raise ValueError('已冻结图片的像素优化模式与本次无像素配置不一致')
        return Path(ready['candidates']['final']['image']), ready

    if (root / 'PAUSE_REQUESTED').exists():
        raise InterruptedError('实验已暂停')

    tick = time.perf_counter()

    # Stage 1: Foreground segmentation
    seg = root / 'foreground_segmentation'
    foreground, _ = segment_owned_image(data['source'], definition['source_boxes'], seg / 'source')
    target_fg, _ = segment_owned_image(data['target'], definition['target_boxes'], seg / 'target')

    # Stage 2: Foreground statistics
    statistics, body = foreground_statistics(data['target'], target_fg)
    stats_root = root / 'foreground_statistics'
    stats_root.mkdir(exist_ok=True)
    Image.fromarray(body.astype(np.uint8) * 255).save(stats_root / 'body_samples.png')
    write_json(stats_root / 'statistics.json', statistics)
    print(json.dumps(dict(stage='foreground_statistics', rgb=statistics['rgb'], method=METHOD)), flush=True)

    # Stage 3: Harmonization with strict zero-pixel assertions
    init = definition.get('initialization', {})
    rgb = statistics['rgb']
    plan = dict(
        instruction=(f'Give only unprinted product surfaces a subtle coherent natural finish inspired by RGB {rgb}. '
                     'Preserve the product shape, every original word, number, drawing and brand mark exactly. Preserve the scene.'),
        editable_regions=[[0, 0, 1, 1]], protected_regions=[], reason='主体统计选色后的统一生成步骤。',
        region_policy='label_protected', feather_pixels=3.0)
    initial_root = root / 'initialization'

    init_opt_cfg = initialization_config(definition)
    assert init_opt_cfg.pixel_clean_steps == 0, 'Stage 3 检查失败：pixel_clean_steps 必须为 0'
    assert init_opt_cfg.pixel_diffusion_steps == 0, 'Stage 3 检查失败：pixel_diffusion_steps 必须为 0'
    assert init_opt_cfg.pixel_levels == 0, 'Stage 3 检查失败：pixel_levels 必须为 0'

    _, initial = prepare_generation_initialization(
        **data, output=initial_root,
        plan=EditPlan(**init.get('plan', plan)), surface_regions=definition['print_regions'],
        config=init_opt_cfg,
        generation=NaturalGenerationConfig(**init.get('generation', {})), device=device,
        pause_path=root / 'PAUSE_REQUESTED')

    init_meta = read(initial_root / 'attack/metadata.json')
    assert init_meta.get('pixel_residual_max_levels', 0) == 0, '初始化残差必须为 0'
    assert not any(row.get('phase', '').startswith('pixel') for row in init_meta.get('history', [])), (
        '初始化优化器历史不得包含像素阶段'
    )

    old_anchor = read_rgb(initial['candidates']['generation_only']['image'])
    old_final = read_rgb(initial['candidates']['natural_final']['image'])
    init_residual = old_final.astype(np.int16) - old_anchor.astype(np.int16)
    assert np.abs(init_residual).max() == 0, '内部起点的辅助残差必须严格为 0'

    from src.visual.encoders.clip import make_encoder
    from src.visual.diffusion_purification import LatentDiffusionPurifier, PurificationConfig
    base = make_encoder(device=device)
    fused, _ = _fuse(data, initial, foreground, statistics, definition['print_regions'], root, config, base)

    fused_anchor = read_rgb(fused / 'generative_anchor.png')
    fused_initial = read_rgb(fused / 'initial.png')
    fused_residual = fused_initial.astype(np.int16) - fused_anchor.astype(np.int16)
    assert np.abs(fused_residual).max() == 0, '颜色融合输出初始图与生成锚点残差必须严格为 0'

    # Stage 4: Latent generation refinement with strict zero-pixel assertions
    start_tensor = read_tensor(fused / 'initial.png', base.surrogate.device)
    g0_tensor = read_tensor(fused / 'generative_anchor.png', base.surrogate.device)
    fixed_pixel = start_tensor - g0_tensor
    assert fixed_pixel.abs().max() == 0, '固定像素残差张量必须严格为 0'
    assert config.latent['auxiliary_levels'] == 0, '潜变量细化辅助残差等级必须为 0'

    purifier = LatentDiffusionPurifier(PurificationConfig(working_size=512), device=device)
    purifier.vae.enable_gradient_checkpointing()
    print('初始化与统计颜色融合完成，开始无像素生成感知优化潜变量细化。', flush=True)
    image, meta = refine_harmonized_generation(
        **data, anchor=fused / 'generative_anchor.png',
        initial=fused / 'initial.png', teacher=initial['candidates']['natural_final']['image'],
        mask_path=fused / 'edit_mask.png', source_foreground=seg / 'source/foreground.png',
        parent_anchor=initial['candidates']['generation_only']['image'], print_regions=definition['print_regions'],
        output=root / 'latent_generation_refinement', surrogate=base.surrogate, purifier=purifier,
        config=PerLAGELatentConfig(**config.latent), pause_path=root / 'PAUSE_REQUESTED')
    if meta['status'] != 'completed':
        raise InterruptedError('无像素方法已保存潜变量细化断点')

    assert meta['auxiliary_max_levels'] == 0, '生成潜变量细化 auxiliary_max_levels 必须严格为 0'
    assert meta['auxiliary_mae_levels'] == 0.0, '生成潜变量细化 auxiliary_mae_levels 必须严格为 0'

    final, anchor = root / 'adversarial.png', root / 'generative_anchor.png'
    shutil.copyfile(image, final)
    shutil.copyfile(root / 'latent_generation_refinement/generative_anchor.png', anchor)

    final_rgb = read_rgb(final)
    anchor_rgb = read_rgb(anchor)
    assert np.array_equal(final_rgb, anchor_rgb), '最终对抗图与生成锚点图必须完全一致（无像素加性扰动）'

    init_identity = initial_root / 'run_identity.json'
    optimizer_identity = initial_root / 'attack/identity.json'
    ready = dict(
        method=METHOD, definition=DEFINITION, status='completed', color_scheme='statistics',
        training_data_policy=TRAINING_DATA_POLICY, shadow_pool_used=False, purchase_history_used=False,
        victim_memory_used=False, formal_queries_used=False, configuration_sha256=config_digest,
        method_identity=str(identity_path), method_identity_sha256=sha256_file(identity_path),
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
        auxiliary_max_levels=0,
        latent_config=config.latent,
        pixel_optimization=False,
        optimization_mode='latent_only',
        source_method='four_natural_stages_with_perceptual_latent_optimization_no_pixel'
    )
    validate_complete(ready, final, method=METHOD)
    write_json(root / 'ready.json', ready)
    del base, purifier
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return final, ready
