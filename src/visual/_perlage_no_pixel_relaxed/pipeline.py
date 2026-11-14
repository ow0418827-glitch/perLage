"""独立约束放宽试验；原无像素方法及其默认配置保持只读。

同组原完整运行的前三阶段和初始化作为明确冻结的公共前缀复用。
本模块只重新执行第四阶段；完成证据同时绑定复用前缀和新细化。
训练输入仍仅限公开图对、源商品名及原冻结的一条目标特点正文。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import gc
import hashlib
import json
import math
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from src.paths import PROJECT, ProjectPath as Path, model_path, relocate_io_path
from src.visual.image_utils import read_tensor, quantize_ste, save_tensor_image, sha256_file, write_json
from src.visual._perlage.components.colour import read_rgb, prepare_print_context, halo_error, print_region_mask
from src.visual._perlage.components.core import source_ssim, tangent_projection
from src.visual._perlage.components.evidence import STAGES, stage_record
from src.visual._perlage.components.latent_branch import SourceLatentBranch
from src.visual._perlage_no_pixel.evidence import validate_complete as validate_original_complete
from src.visual._perlage_no_pixel.pipeline import validate_definition
from src.visual.retrieval_objective import JointRetrievalObjective, TRAINING_DATA_POLICY, training_data
from src.visual.robust_objectives import softmin


METHOD = 'PerLAGE_no_pixel_relaxed_experiment'
DEFINITION = 'frozen_three_stage_prefix_absolute_joint_latent_relaxation_v1'


@dataclass(frozen=True)
class RelaxedLatentConfig:
    steps: int = 160
    check_every: int = 10
    latent_levels: int = 64
    latent_working_size: int = 256
    latent_limit: float = 1.0
    latent_learning_rate: float = 0.02
    latent_smoothing_mix: float = 0.06
    source_ssim_floor: float = 0.75
    source_mae_ceiling: float = 24.0
    temperature: float = 0.04
    macro_l2_weight: float = 0.00005
    macro_tv_weight: float = 0.0003
    halo_parent_multiplier: float = 1.0
    halo_floor: float = 12.0
    guard_backtracks: int = 8
    auxiliary_levels: int = 0
    feature_weight: float = 0.0
    diffusion_weight: float = 0.0
    seed: int = 42

    def validate(self):
        if type(self.steps) is not int or self.steps < 1 or type(self.check_every) is not int or self.check_every < 1 or self.steps % self.check_every:
            raise ValueError('试验须执行完整、可检查的细化步数')
        if type(self.latent_levels) is not int or not 1 <= self.latent_levels <= 128:
            raise ValueError('试验潜变量解码变化预算无效')
        if not 0.5 <= self.source_ssim_floor < 1 or not 0 < self.source_mae_ceiling <= 64:
            raise ValueError('试验质量门槛无效')
        if not 0 < self.latent_limit <= 4 or not 0 < self.latent_learning_rate <= 0.1 or not 0 <= self.latent_smoothing_mix <= 1:
            raise ValueError('试验潜变量更新参数无效')
        if self.auxiliary_levels != 0 or self.feature_weight != 0 or self.diffusion_weight != 0:
            raise ValueError('本试验禁止像素残差、教师特征保持及扩散净化目标')
        if not 0 < self.temperature <= 1 or min(self.macro_l2_weight, self.macro_tv_weight) < 0:
            raise ValueError('试验目标参数无效')
        if self.halo_floor < 0 or self.halo_parent_multiplier < 0 or type(self.guard_backtracks) is not int or not 0 <= self.guard_backtracks <= 16:
            raise ValueError('试验自然度守门参数无效')
        if type(self.seed) is not int or self.seed < 0 or self.latent_working_size < 64:
            raise ValueError('试验种子或编码尺寸无效')
        if any(not math.isfinite(v) for v in asdict(self).values() if isinstance(v, float)):
            raise ValueError('试验参数必须有限')


RELAXATION_PRESETS = {
    'relaxed': asdict(RelaxedLatentConfig()),
    'strong': asdict(RelaxedLatentConfig(
        steps=240, latent_levels=96, latent_limit=2.0,
        latent_learning_rate=0.03, source_ssim_floor=0.65,
        source_mae_ceiling=40.0)),
}


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _canonical(value):
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _files(paths):
    result = {}
    for value in paths:
        path = Path(value).resolve()
        digest = sha256_file(path)
        if not digest:
            raise FileNotFoundError('缺少绑定文件：' + str(path))
        result[str(path)] = digest
    return result


def validate_prefix(prefix_root, definition):
    """只读核对本项目同组原完整前缀；不读取正式查询或记忆库。"""
    validate_definition(definition)
    data = training_data(definition)
    root = Path(prefix_root).resolve()
    root.relative_to(PROJECT.resolve())
    original = validate_original_complete(_read(root / 'ready.json'))
    if not original.get('method_identity_sha256') or sha256_file(original.get('method_identity')) != original['method_identity_sha256']:
        raise ValueError('复用前缀缺少真实方法身份摘要')
    for role in ('source', 'target'):
        if original[role + '_sha256'] != sha256_file(data[role]):
            raise ValueError('复用前缀的公开图对不一致：' + role)
    title_digest = hashlib.sha256(data['source_title'].encode('utf-8')).hexdigest()
    if original['source_title_sha256'] != title_digest:
        raise ValueError('复用前缀的公开源商品名改变')
    initial = _read(original['initialization_identity'])
    if initial['training_queries'] != data['training_queries'] or initial['surface_regions'] != definition['print_regions']:
        raise ValueError('复用前缀的训练正文或印刷保护区改变')
    for role in ('source', 'target'):
        segmentation = _read(root / 'foreground_segmentation' / role / 'record.json')
        if segmentation['identity']['boxes'] != definition[role + '_boxes']:
            raise ValueError('复用前缀的分割提示改变：' + role)
    init_opt = _read(original['initialization_optimizer_identity'])
    if any(init_opt['config'][key] != 0 for key in ('pixel_clean_steps', 'pixel_diffusion_steps', 'pixel_levels')):
        raise ValueError('复用前缀存在像素优化')
    anchor = root / 'color_harmonization/generative_anchor.png'
    initial_image = root / 'color_harmonization/initial.png'
    if not np.array_equal(read_rgb(anchor), read_rgb(initial_image)):
        raise ValueError('复用前缀存在固定像素残差')
    init_ready = _read(root / 'initialization/ready.json')
    teacher = Path(init_ready['candidates']['natural_final']['image'])
    parent_anchor = Path(init_ready['candidates']['generation_only']['image'])
    for key, record in init_ready['candidates'].items():
        if sha256_file(record['image']) != record['sha256']:
            raise ValueError('复用初始化图像改变：' + key)
    binding = {
        'root': str(root), 'ready_sha256': sha256_file(root / 'ready.json'),
        'first_three_stages_reused': True, 'fresh_first_three_stages_executed': False,
        'initialization_reused': True, 'previous_final_refinement_reused': False,
        'stages': {name: copy.deepcopy(original['stages'][name]) for name in STAGES[:3]},
        'initialization_evidence': original['initialization_evidence'],
        'files': _files([
            root / 'ready.json', original['initialization_identity'],
            original['initialization_optimizer_identity'], root / 'initialization/ready.json',
            anchor, initial_image, teacher, parent_anchor,
            root / 'color_harmonization/edit_mask.png',
            root / 'foreground_segmentation/source/foreground.png']),
    }
    return root, original, data, teacher, parent_anchor, _canonical(binding)


class _VaeOnly:
    """同一冻结自编码器，只提供解码路线，不加载或调用噪声网络。"""
    def __init__(self, device):
        from diffusers import AutoencoderKL
        self.device = torch.device(device)
        root = Path(model_path('stable-diffusion-v1-5')).resolve()
        self.vae = AutoencoderKL.from_pretrained(
            str(root), subfolder='vae', local_files_only=True,
            use_safetensors=True, variant='fp16', torch_dtype=torch.float32)
        self.vae.eval().requires_grad_(False).to(self.device)
        self.vae.enable_gradient_checkpointing()
        self.identity = {
            'kind': 'same_sd15_vae_only_no_noise_network',
            'files': _files(p for p in sorted((root / 'vae').glob('*'))
                            if p.is_file() and p.suffix in {'.json', '.safetensors', '.bin'}),
            'precision': 'float32', 'posterior': 'mode', 'diffusion_calls': 0,
        }

    def _decode(self, latent, shape):
        image = (self.vae.decode(latent / self.vae.config.scaling_factor).sample.float() / 2 + 0.5).clamp(0, 1)
        return F.interpolate(image, size=shape[-2:], mode='bicubic', align_corners=False, antialias=True).clamp(0, 1)


def validate_complete(ready, image=None):
    if ready.get('method') != METHOD or ready.get('definition') != DEFINITION or ready.get('status') != 'completed':
        raise ValueError('独立放宽试验尚未完整完成')
    if ready.get('pixel_optimization') is not False or ready.get('auxiliary_max_levels') != 0:
        raise ValueError('独立放宽试验出现像素优化或残差')
    for name in ('victim_memory_used', 'formal_queries_used', 'shadow_pool_used', 'true_rank_used', 'purchase_history_used'):
        if ready.get(name) is not False:
            raise ValueError('独立放宽试验违反训练边界：' + name)
    if ready.get('first_three_stages_reused') is not True or ready.get('fresh_first_three_stages_executed') is not False:
        raise ValueError('未如实记录冻结前三阶段复用')
    stages = ready.get('stages', {})
    if set(stages) != set(STAGES):
        raise ValueError('缺少四阶段完整证据')
    for stage in stages.values():
        if stage.get('status') != 'completed' or not stage.get('files'):
            raise ValueError('阶段未完成')
        for path, digest in stage['files'].items():
            if not digest or sha256_file(path) != digest:
                raise ValueError('阶段证据改变：' + path)
    last = stages['latent_generation_refinement']
    if type(last.get('steps_completed')) is not int or last['steps_completed'] < 1 or last['steps_completed'] != last.get('steps_requested'):
        raise ValueError('未完成全部细化步数')
    identity_path = ready['method_identity']
    if not ready.get('method_identity_sha256') or sha256_file(identity_path) != ready['method_identity_sha256']:
        raise ValueError('方法身份改变')
    identity = _read(identity_path)
    if identity.get('method') != METHOD or identity.get('definition') != DEFINITION or identity.get('profile') != ready.get('profile'):
        raise ValueError('完成记录与独立方法身份不一致')
    config = identity['config']
    RelaxedLatentConfig(**config).validate()
    if ready.get('latent_config') != config or last['steps_requested'] != config['steps']:
        raise ValueError('完成记录的档位配置或步数改变')
    if ready.get('prefix_binding') != identity['prefix_binding']:
        raise ValueError('完成记录的公共前缀绑定改变')
    if ready.get('training_data_policy') != TRAINING_DATA_POLICY:
        raise ValueError('完成记录的训练输入策略改变')
    public_data = training_data(identity['public_definition'])
    for role in ('source', 'target'):
        path = str(Path(public_data[role]).resolve())
        expected = identity['inputs'].get(path)
        if not expected or expected != ready.get(role + '_sha256') or sha256_file(path) != expected:
            raise ValueError('完成记录的公开图对绑定改变：' + role)
    title_digest = hashlib.sha256(public_data['source_title'].encode('utf-8')).hexdigest()
    if ready.get('source_title_sha256') != title_digest:
        raise ValueError('完成记录的公开源商品名绑定改变')
    if identity.get('pixel_optimization') is not False or identity.get('auxiliary_levels') != 0 or identity.get('diffusion_calls') != 0:
        raise ValueError('方法身份违反零像素及无净化条件')
    for path, digest in identity['prefix_binding']['files'].items():
        if sha256_file(path) != digest:
            raise ValueError('冻结公共前缀改变')
    candidate = ready['candidates']['final']
    if not candidate.get('sha256') or sha256_file(candidate['image']) != candidate['sha256'] or (image is not None and sha256_file(image) != candidate['sha256']):
        raise ValueError('最终图片摘要改变')
    if not np.array_equal(read_rgb(candidate['image']), read_rgb(ready['generative_anchor'])):
        raise ValueError('最终图片存在像素加性残差')
    return ready


def generate_relaxed_no_pixel(source, target, output, *, definition, prefix_root,
                              profile='relaxed', device='cuda', pause_path=None):
    """按预注册档位执行完整第四阶段，返回图片及四阶段完成记录。

    输出目录绑定输入、前缀、档位、编码器、自编码器及代码摘要；中断后
    使用同一调用恢复。不会根据正式排名选择档位或重新挑选图片。
    """
    if profile not in RELAXATION_PRESETS:
        raise ValueError('未知的预定义约束档位')
    config = RelaxedLatentConfig(**RELAXATION_PRESETS[profile])
    config.validate()
    prefix, original_ready, data, teacher, parent_anchor, prefix_binding = validate_prefix(prefix_root, definition)
    for role, given in (('source', source), ('target', target)):
        if sha256_file(given) != sha256_file(data[role]):
            raise ValueError('调用图对和冻结公开配置不一致')
    root = Path(output).resolve()
    root.relative_to(PROJECT.resolve())
    root.mkdir(parents=True, exist_ok=True)
    tick = time.perf_counter()
    pause_path = Path(pause_path) if pause_path else root / 'PAUSE_REQUESTED'
    if pause_path.exists():
        raise InterruptedError('独立放宽试验已暂停')
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    from src.visual.encoders.clip import make_encoder
    base = make_encoder(device=device)
    surrogate = base.surrogate
    device = surrogate.device
    vae = _VaeOnly(device)
    implementation_paths = [Path(__file__), Path(__file__).with_name('__init__.py')]
    implementation_paths += [PROJECT / p for p in (
        'src/visual/_perlage/components/latent_branch.py',
        'src/visual/_perlage/components/core.py', 'src/visual/_perlage/components/colour.py',
        'src/visual/retrieval_objective.py', 'src/visual/robust_objectives.py',
        'src/visual/image_utils.py', 'src/visual/encoders/backbone.py',
        'src/visual/encoders/clip.py', 'src/visual/encoders/joint.py')]
    identity = _canonical(dict(
        method=METHOD, definition=DEFINITION, profile=profile, config=asdict(config),
        public_definition=definition, prefix_binding=prefix_binding,
        inputs=_files([source, target]), encoder=surrogate.snapshot(), vae=vae.identity,
        implementation=_files(implementation_paths),
        training_data_policy=TRAINING_DATA_POLICY, training_queries=data['training_queries'],
        objective='maximize_unpurified_absolute_joint_mean_and_softmin_minus_l2_tv',
        image_text_training_weight=[0.5, 0.5], feature_retention=False,
        ordinary_teacher_score_floor=False, noise_retention_gate=False,
        diffusion_calls=0, pixel_optimization=False, auxiliary_levels=0,
        victim_memory_used=False, formal_queries_used=False, shadow_pool_used=False,
        true_rank_used=False, selection_rule='best_public_training_objective_with_fixed_quality_gates'))
    identity_path = root / 'method_identity.json'
    if identity_path.exists():
        if _read(identity_path) != identity:
            raise ValueError('独立试验身份改变，须使用新目录')
    elif any(root.iterdir()):
        raise ValueError('拒绝覆盖没有独立身份的非空目录')
    else:
        write_json(identity_path, identity)
    if (root / 'ready.json').exists():
        ready = validate_complete(_read(root / 'ready.json'))
        del base, vae
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return Path(ready['candidates']['final']['image']), ready

    original = read_tensor(source, device)
    anchor = read_tensor(prefix / 'color_harmonization/generative_anchor.png', device)
    oldmask = np.asarray(Image.open(relocate_io_path(prefix / 'color_harmonization/edit_mask.png')).convert('L')).copy()
    foreground = read_rgb(prefix / 'foreground_segmentation/source/foreground.png')[..., 0] > 0
    roi = print_region_mask(read_rgb(source).shape, definition['print_regions'])
    macro_array = oldmask.astype(float) / 255 * foreground * ~roi
    macro = torch.as_tensor(macro_array, device=device, dtype=torch.float32)[None, None]
    Image.fromarray(np.rint(macro_array * 255).astype(np.uint8)).save(root / 'latent_edit_mask.png')
    context = prepare_print_context(read_rgb(source), read_rgb(parent_anchor), oldmask, foreground, definition['print_regions'])
    parent_halo = halo_error(context, read_rgb(teacher))
    halo_ceiling = max(parent_halo * config.halo_parent_multiplier, config.halo_floor)
    with torch.no_grad():
        target_feature = surrogate.encode_image(read_tensor(target, device)).global_embedding.detach()
    joint = JointRetrievalObjective(surrogate, target_feature, data['source_title'], data['training_queries'])
    write_json(root / 'public_title_encoding_audit.json', joint.audit)
    branch = SourceLatentBranch(anchor, macro, vae, config)
    optimizer = torch.optim.Adam([branch.delta], lr=config.latent_learning_rate)

    def compose():
        return quantize_ste((anchor + branch.residual()).clamp(0, 1))

    def value_and_penalty(image):
        scores = joint.scores(surrogate.encode_image(image).global_embedding)
        utility = 0.5 * scores.mean() + 0.5 * softmin(scores.flatten(), config.temperature)
        unit = (image - anchor) / (config.latent_levels / 255)
        tv = (unit[..., 1:, :] - unit[..., :-1, :]).abs().mean() + (unit[..., :, 1:] - unit[..., :, :-1]).abs().mean()
        penalty = config.macro_l2_weight * unit.square().mean() + config.macro_tv_weight * tv
        return utility, penalty, scores

    @torch.no_grad()
    def quality(image):
        ss = float(source_ssim(image, original))
        mae = float((image - original).abs().mean() * 255)
        arr = np.rint(image[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        halo = halo_error(context, arr)
        return dict(source_ssim=ss, source_mae_levels=mae, halo_error=halo,
                    quality_ok=ss >= config.source_ssim_floor and mae <= config.source_mae_ceiling,
                    natural_halo_gate=halo <= halo_ceiling)

    @torch.no_grad()
    def development(image, step):
        row = quality(image)
        utility, penalty, scores = value_and_penalty(image)
        row.update(step=step, joint_scores=scores.cpu().tolist(), training_utility=float(utility),
                   regularization=float(penalty), selection_utility=float(utility - penalty),
                   halo_ceiling=halo_ceiling, parent_halo_error=parent_halo)
        row['selection'] = [int(row['quality_ok'] and row['natural_halo_gate']), row['selection_utility']]
        return row

    history, best_row, best = [], None, None
    step_done, previous, accepted, rejected = 0, 0.0, 0, 0
    checkpoint_path = root / 'checkpoint.pt'
    if checkpoint_path.exists():
        state = torch.load(relocate_io_path(checkpoint_path), map_location=device, weights_only=False)
        if state['identity'] != identity:
            raise ValueError('独立试验断点身份改变')
        with torch.no_grad():
            branch.delta.copy_(state['latent'])
        optimizer.load_state_dict(state['optimizer'])
        history, best_row, best = state['history'], state['best_row'], state['best']
        step_done, previous = state['step'], state['seconds']
        accepted, rejected = state['accepted'], state['rejected']

    def save(status):
        elapsed = previous + time.perf_counter() - tick
        pending = checkpoint_path.with_suffix('.tmp')
        torch.save(dict(identity=identity, latent=branch.delta.detach(), optimizer=optimizer.state_dict(),
                        history=history, best_row=best_row, best=best, step=step_done,
                        seconds=elapsed, accepted=accepted, rejected=rejected), relocate_io_path(pending))
        pending.replace(checkpoint_path)
        write_json(root / 'progress.json', dict(status=status, profile=profile, step=step_done,
                   requested_steps=config.steps, best=best_row, accepted_updates=accepted,
                   rejected_updates=rejected, seconds=elapsed))

    if best_row is None:
        best = branch.delta.detach().clone()
        best_row = development(compose(), 0)
        if best_row['selection'][0] != 1:
            write_json(root / 'initial_quality_rejection.json', best_row)
            raise ValueError('冻结起点未通过本轮固定质量守门')
        history.append(best_row)
        save('running')
    for step in range(step_done + 1, config.steps + 1):
        if pause_path.exists():
            save('paused')
            raise InterruptedError('独立放宽试验已保存细化断点')
        image = compose().detach().requires_grad_(True)
        normals, slacks = [], []
        structure = source_ssim(image, original)
        normal, = torch.autograd.grad(structure, image)
        normals.append(normal.detach())
        slacks.append(float(structure.detach()) - config.source_ssim_floor)
        average = (image - original).abs().mean()
        normal, = torch.autograd.grad(-average, image)
        normals.append(normal.detach())
        slacks.append(config.source_mae_ceiling / 255 - float(average.detach()))
        utility, penalty, _ = value_and_penalty(image)
        gradient, = torch.autograd.grad(-utility + penalty, image)
        if not torch.isfinite(gradient).all():
            raise FloatingPointError('独立试验图像梯度非有限')
        old = branch.delta.detach().clone()
        old_optimizer = copy.deepcopy(optimizer.state_dict())
        optimizer.zero_grad(set_to_none=True)
        decoded = compose()
        latent_normals = []
        for normal in normals:
            derivative, = torch.autograd.grad(decoded, branch.delta, grad_outputs=normal, retain_graph=True)
            latent_normals.append(derivative.detach())
        decoded.backward(gradient)
        if branch.delta.grad is None or not torch.isfinite(branch.delta.grad).all():
            raise FloatingPointError('独立试验潜变量梯度非有限')
        optimizer.param_groups[0]['lr'] = config.latent_learning_rate * (1 - 0.75 * step / config.steps)
        optimizer.step()
        with torch.no_grad():
            branch.delta.clamp_(-config.latent_limit, config.latent_limit)
            smooth = F.avg_pool2d(F.pad(branch.delta, (1, 1, 1, 1), mode='reflect'), 3, stride=1)
            branch.delta.lerp_(smooth, config.latent_smoothing_mix)
            projected = tangent_projection(branch.delta - old, latent_normals, slacks)
            proposed = (old + projected).clamp(-config.latent_limit, config.latent_limit)
            fraction = 0.0
            for backtrack in range(config.guard_backtracks + 1):
                scale = 0.5 ** backtrack
                branch.delta.copy_(old + scale * (proposed - old))
                q = quality(compose())
                if q['quality_ok'] and q['natural_halo_gate']:
                    fraction = scale
                    break
            if fraction:
                accepted += 1
            else:
                branch.delta.copy_(old)
                optimizer.load_state_dict(old_optimizer)
                rejected += 1
        step_done = step
        if step % config.check_every == 0:
            row = development(compose(), step)
            history.append(row)
            save_tensor_image(root / f'checkpoint_{step:04d}.png', compose())
            if tuple(row['selection']) > tuple(best_row['selection']):
                best, best_row = branch.delta.detach().clone(), row
            save('running')
            print(json.dumps(dict(profile=profile, step=step, best_step=best_row['step'],
                  utility=row['training_utility'], source_ssim=row['source_ssim'],
                  source_mae_levels=row['source_mae_levels'], accepted=accepted,
                  rejected=rejected), ensure_ascii=False), flush=True)
    with torch.no_grad():
        branch.delta.copy_(best)
        final_tensor = compose()
        final_check = development(final_tensor, best_row['step'])
    if final_check['selection'][0] != 1:
        raise AssertionError('最终图像未通过固定质量守门')
    final = root / 'adversarial.png'
    generative_anchor = root / 'generative_anchor.png'
    save_tensor_image(final, final_tensor)
    save_tensor_image(generative_anchor, final_tensor)
    final_rgb, anchor_rgb = read_rgb(final), read_rgb(prefix / 'color_harmonization/generative_anchor.png')
    change = np.abs(final_rgb.astype(np.int16) - anchor_rgb.astype(np.int16))
    if change.max() > config.latent_levels or np.any(change[roi]) or np.any(change[~foreground]):
        raise AssertionError('潜变量预算、前景或印刷区域保护失败')
    metadata = dict(status='completed', config=asdict(config), history=history,
        steps_completed=step_done, best=best_row, final_check=final_check,
        accepted_updates=accepted, rejected_updates=rejected,
        latent_max_levels=int(change.max()), latent_mae_levels=float(change.mean()),
        auxiliary_max_levels=0, auxiliary_mae_levels=0.0, pixel_optimization=False,
        no_pixel_optimization=True, diffusion_calls=0, printed_region_fixed_to_harmonized_anchor=True,
        foreground_limited=True, private_gallery_used=False, true_rank_used=False,
        elapsed_seconds=previous + time.perf_counter() - tick,
        output_sha256=sha256_file(final))
    write_json(root / 'metadata.json', metadata)
    stages = copy.deepcopy(prefix_binding['stages'])
    for record in stages.values():
        record['execution_mode'] = 'reused_frozen_prefix'
    stages[STAGES[-1]] = stage_record([identity_path, root / 'metadata.json', final,
        generative_anchor, root / 'latent_edit_mask.png', root / 'public_title_encoding_audit.json'],
        steps_completed=step_done, steps_requested=config.steps, execution_mode='fresh_full_refinement')
    ready = dict(method=METHOD, definition=DEFINITION, profile=profile, status='completed',
        color_scheme='statistics', training_data_policy=TRAINING_DATA_POLICY,
        first_three_stages_reused=True, fresh_first_three_stages_executed=False,
        initialization_reused=True, previous_final_refinement_reused=False,
        method_identity=str(identity_path), method_identity_sha256=sha256_file(identity_path),
        source_sha256=sha256_file(source), target_sha256=sha256_file(target),
        source_title_sha256=original_ready['source_title_sha256'],
        stages=stages, latent_config=asdict(config),
        candidates={'final': {'image': str(final), 'sha256': sha256_file(final)}},
        generative_anchor=str(generative_anchor), prefix_binding=prefix_binding,
        pixel_optimization=False, auxiliary_max_levels=0, optimization_mode='latent_only',
        victim_memory_used=False, formal_queries_used=False, shadow_pool_used=False,
        true_rank_used=False, purchase_history_used=False, diffusion_calls=0,
        observed_seconds=metadata['elapsed_seconds'])
    validate_complete(ready, final)
    write_json(root / 'ready.json', ready)
    save('completed')
    del base, vae, surrogate, branch, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return final, ready
