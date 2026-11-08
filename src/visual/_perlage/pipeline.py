"""独立自然优先感知优化对抗方法 (PerLAGE) 的四阶段完整管线。"""
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
    HarmonizationConfig, foreground_statistics, segment_owned_image)
from src.visual._perlage.components.assets import NaturalGenerationConfig
from src.visual._perlage.components.evidence import stage_record
from src.visual._perlage.components.optimizer import InitializationOptimizationConfig
from src.visual._perlage.initialization import prepare_generation_initialization
from src.visual._perlage.latent import PerLAGELatentConfig, refine_harmonized_generation
from src.visual._perlage.evidence import (
    METHOD, DEFINITION, NO_PIXEL_METHOD, NO_PIXEL_DEFINITION,
    definition_for_method, validate_complete, read)
from src.visual.image_utils import sha256_file, write_json
from src.visual._perlage.components.instruction import EditPlan
from src.visual._perlage.harmonization import _fuse
from src.visual.retrieval_objective import TRAINING_DATA_POLICY, training_data


@dataclass(frozen=True)
class PerLAGEConfig:
    harmonization: dict = field(default_factory=lambda: dict(
        smoothing_sigma=12.0, print_ring_sigma=7.0, selected_color_mix=0.25,
        strength=0.75, blend_mode='matte'))
    latent: dict = field(default_factory=lambda: dict(asdict(PerLAGELatentConfig()), auxiliary_levels=0))
    pixel_optimization: bool = False


def validate_definition(definition, *, method=None):
    if method is None:
        method = definition.get('method', METHOD)
    if definition.get('method') != method or definition.get('definition') != definition_for_method(method):
        raise ValueError('优化方法必须使用独立的方法身份，不能作为原方法的分支')
    if any(key in definition for key in ('public_pool', 'completed_asset', 'interface', 'variant', 'study_arm')):
        raise ValueError('优化方法禁止竞争池、旧成图与分支选择')
    if definition.get('color_scheme') != 'statistics':
        raise ValueError('优化方法仍保留主体统计选色')
    training_data(definition)
    initial = definition.get('initialization', {})
    if set(initial) - {'optimizer', 'generation', 'plan'}:
        raise ValueError('初始化须在当前输出目录重新计算，不接受其他成图路径')
    optimizer = initialization_config(definition, method=method)
    if initial.get('optimizer', {}).get('shadow_weight', 0) != 0:
        raise ValueError('优化方法初始化不能启用竞争损失')
    optimizer.validate()
    defaults = PerLAGEConfig()
    expected_pixel = method != NO_PIXEL_METHOD and definition.get('pixel_optimization', False)
    pixel_optimization = definition.get('pixel_optimization', expected_pixel)
    if type(pixel_optimization) is not bool:
        raise ValueError('像素优化开关必须为布尔值')
    if method == NO_PIXEL_METHOD and pixel_optimization:
        raise ValueError('无像素方法禁止开启像素优化')
    latent = dict(defaults.latent, **definition.get('latent', {}))
    if pixel_optimization:
        if 'auxiliary_levels' not in definition.get('latent', {}):
            latent['auxiliary_levels'] = 4
        elif latent['auxiliary_levels'] <= 0:
            raise ValueError('含像素感知方法不能关闭辅助像素预算')
    else:
        if definition.get('latent', {}).get('auxiliary_levels', 0) != 0:
            raise ValueError('仅潜变量模式禁止像素优化：固定像素残差')
        latent['auxiliary_levels'] = 0
    config = PerLAGEConfig(dict(defaults.harmonization, **definition.get('harmonization', {})),
                          latent, pixel_optimization)
    colour = HarmonizationConfig(**config.harmonization)
    colour.validate()
    if colour.blend_mode != 'matte':
        raise ValueError('保持相同的线性光颜色融合')
    PerLAGELatentConfig(**config.latent).validate()
    return config


def generate_candidate(source, target, output, *, definition, device='cuda', method=METHOD):
    config = validate_definition(definition, method=method)
    data = training_data(definition)
    for key, given in (('source', source), ('target', target)):
        if sha256_file(given) != sha256_file(data[key]):
            raise ValueError('配置和当前公开图对不符')
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    config_digest = hashlib.sha256(json.dumps(definition, sort_keys=True).encode('utf-8')).hexdigest()
    identity_path = root / 'method_identity.json'
    implementations = ['src/visual/_perlage/pipeline.py', 'src/visual/_perlage/evidence.py',
        'src/visual/_perlage/initialization.py', 'src/visual/_perlage/latent.py',
        'src/visual/_perlage/components/optimizer.py', 'src/visual/_perlage/components/core.py',
        'src/visual/_perlage/components/assets.py', 'src/visual/_perlage/components/editor.py',
        'src/visual/_perlage/components/instruction.py', 'src/visual/_perlage/components/mask.py',
        'src/visual/_perlage/components/colour.py', 'src/visual/_perlage/components/segmentation.py',
        'src/visual/_perlage/components/losses.py', 'src/visual/_perlage/components/latent_branch.py',
        'src/visual/_perlage/harmonization.py', 'src/visual/encoders/backbone.py',
        'src/visual/encoders/clip.py', 'src/visual/encoders/joint.py',
        'src/visual/retrieval_objective.py', 'src/visual/robust_objectives.py',
        'src/visual/diffusion_components.py', 'src/visual/diffusion_purification.py',
        'src/visual/config/optimization.json', 'src/model_identity.py']
    identity = dict(method=method, definition=definition_for_method(method),
        configuration_sha256=config_digest, source_sha256=sha256_file(source), target_sha256=sha256_file(target),
        implementation={name: sha256_file(PROJECT / name) for name in implementations})
    if identity_path.exists():
        if read(identity_path) != identity:
            raise ValueError('独立方法、输入或实现已改变，请使用新的输出目录')
    elif any(root.iterdir()):
        raise ValueError('非空目录没有本次独立方法身份，禁止复用旧图或旧断点')
    else:
        write_json(identity_path, identity)
    if (root / 'ready.json').exists():
        ready = validate_complete(read(root / 'ready.json'), method=method)
        if ready['configuration_sha256'] != config_digest:
            raise ValueError('冻结配置改变，请另建实验目录')
        if ready.get('pixel_optimization', True) != config.pixel_optimization:
            raise ValueError('已冻结图片的像素优化模式与本次配置不一致')
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
    print(json.dumps(dict(stage='foreground_statistics', rgb=statistics['rgb'], method=method)), flush=True)
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
        config=initialization_config(definition, method=method),
        generation=NaturalGenerationConfig(**init.get('generation', {})), device=device,
        pause_path=root / 'PAUSE_REQUESTED')
    from src.visual.encoders.clip import make_encoder
    from src.visual.diffusion_purification import LatentDiffusionPurifier, PurificationConfig
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
        method=method, definition=definition_for_method(method), status='completed', color_scheme='statistics',
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
        auxiliary_max_levels=meta['auxiliary_max_levels'],
        latent_config=config.latent,
        pixel_optimization=config.pixel_optimization,
        optimization_mode='latent_and_pixel' if config.pixel_optimization else 'latent_only',
        source_method='four_natural_stages_with_perceptual_latent_optimization'
    )
    validate_complete(ready, final, method=method)
    write_json(root / 'ready.json', ready)
    del base, purifier
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return final, ready


def initialization_config(definition, *, method=None):
    """感知方法根据 method 与 pixel_optimization 开关配置是否执行像素阶段。"""
    if method is None:
        method = definition.get('method', METHOD)
    if definition.get('method') != method or definition.get('definition') != definition_for_method(method):
        raise ValueError('初始化配置与独立方法身份不一致')
    pixel_optimization = method != NO_PIXEL_METHOD and definition.get('pixel_optimization', False)
    values = dict(definition.get('initialization', {}).get('optimizer', {}), shadow_weight=0.0)
    if not pixel_optimization:
        for key in ('pixel_clean_steps', 'pixel_diffusion_steps', 'pixel_levels'):
            if values.get(key, 0) != 0:
                raise ValueError('仅潜变量模式不能配置像素优化：' + key)
            values[key] = 0
    else:
        if 'pixel_clean_steps' not in values:
            values['pixel_clean_steps'] = 20
        if 'pixel_diffusion_steps' not in values:
            values['pixel_diffusion_steps'] = 60
        if 'pixel_levels' not in values:
            values['pixel_levels'] = 4
    config = InitializationOptimizationConfig(**values)
    config.validate()
    if pixel_optimization and any(getattr(config, key) <= 0 for key in (
            'pixel_clean_steps', 'pixel_diffusion_steps', 'pixel_levels')):
        raise ValueError('完整感知方法必须保留两段像素阶段及像素预算')
    return config


def _generate_perlage(source, target, output, *, definition, device='cuda', method=METHOD):
    """从公开图对、名称和几何提示生成完整优化对抗图片，接口不读取实验或私有评估。"""
    data = training_data(definition)
    image, ready = generate_candidate(source, target, output,
                                          definition=definition, device=device, method=method)
    validate_complete(ready, image, method=method)
    for name, path in (('source', source), ('target', target)):
        if ready[name + '_sha256'] != sha256_file(path):
            raise ValueError('已有优化对抗成图的公开输入改变：' + name)
    title_sha = hashlib.sha256(data['source_title'].encode('utf-8')).hexdigest()
    if ready['source_title_sha256'] != title_sha:
        raise ValueError('已有优化对抗成图不属于当前公开商品名')
    return Path(image), ready


def generate_perlage(source, target, output, *, definition, device='cuda'):
    """感知方法入口；根据 definition 中的 pixel_optimization 执行。"""
    return _generate_perlage(source, target, output, definition=definition, device=device, method=METHOD)


def generate_perlage_no_pixel(source, target, output, *, definition, device='cuda'):
    """独立无像素扰动感知方法入口。"""
    no_pixel_def = dict(definition, method=NO_PIXEL_METHOD, definition=NO_PIXEL_DEFINITION, pixel_optimization=False)
    no_pixel_def.setdefault('latent', {})['auxiliary_levels'] = 0
    return _generate_perlage(source, target, output, definition=no_pixel_def, device=device, method=NO_PIXEL_METHOD)


def generate_perlage_latent_only(source, target, output, *, definition, device='cuda'):
    """仅潜变量优化入口别名。"""
    return generate_perlage_no_pixel(source, target, output, definition=definition, device=device)
