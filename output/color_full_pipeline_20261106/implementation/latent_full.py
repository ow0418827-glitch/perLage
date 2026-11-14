"""独立自然优先感知优化对抗方法 (PerLAGE) 的潜变量细化模块。"""
from dataclasses import dataclass, asdict
import copy
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from src.paths import relocate_io_path
from src.vision.image_utils import read_tensor, quantize_ste, save_tensor_image, sha256_file, write_json
from src.vision._natural.core import pixel_projection, source_ssim, tangent_projection
from src.vision.latent_branch import SourceLatentBranch
from src.vision.robust_objectives import softmin, transformed_batch
from src.vision.retrieval_objective import JointRetrievalObjective, TRAINING_DATA_POLICY
from src.vision._natural.losses import owned_retention_loss
from src.vision._natural.colour import read_rgb, prepare_print_context, halo_error
from src.vision._natural.colour import print_region_mask as region_mask


@dataclass(frozen=True)
class PerLAGELatentConfig:
    steps: int = 60
    check_every: int = 10
    latent_levels: int = 24
    latent_working_size: int = 256
    latent_limit: float = 0.5
    latent_learning_rate: float = 0.010
    latent_smoothing_mix: float = 0.12
    auxiliary_levels: int = 4
    source_ssim_floor: float = 0.85
    source_mae_ceiling: float = 12.0
    ordinary_tolerance: float = 0.01
    temperature: float = 0.01
    desired_owned_score_gain: float = 0.02
    feature_weight: float = 0.02
    diffusion_weight: float = 3.0
    macro_l2_weight: float = 0.001
    macro_tv_weight: float = 0.003
    guard_backtracks: int = 8
    full_eot_samples: int = 2
    training_seed_base: int = 720300000
    timesteps: tuple = (99, 199, 399)
    development_seeds: tuple = (9401, 9402, 9403, 9404, 9405, 9406, 9407, 9408)
    selection_seeds: tuple = (9301, 9302)
    retention_tolerance: float = 0.0005

    def validate(self):
        if self.steps < 1 or self.check_every < 1 or self.steps % self.check_every:
            raise ValueError('生成步数需要完整检查点')
        if not 0 < self.latent_levels <= 32 or not 0 <= self.auxiliary_levels <= 4:
            raise ValueError('生成/辅助预算无效')
        if not 0 < self.latent_learning_rate or not 0 < self.latent_limit or (not 0 <= self.latent_smoothing_mix <= 1):
            raise ValueError('潜变量配置无效')
        if self.full_eot_samples not in (1, 2) or len(self.development_seeds) < 4:
            raise ValueError('噪声开发面板或训练样本数无效')
        if not 0.75 <= self.source_ssim_floor <= 0.99 or not 0 < self.source_mae_ceiling <= 16:
            raise ValueError('质量门槛无效')


def pool_free_value(scores, temperature):
    """平衡平均分与最弱训练请求的绝对联合分数。"""
    if scores.ndim != 2 or not torch.isfinite(scores).all():
        raise ValueError("开发分数必须是有限二维张量")
    return 0.5 * scores.mean() + 0.5 * softmin(scores.flatten(), temperature)


def relative_noise_panel(gaps, timesteps):
    """每条路径先取开发请求最弱分数差，每档再取下四分位数。"""
    result = {int(t): float(np.quantile([r['minimum_query_gap'] for r in gaps if r['timestep'] == t], 0.25)) for t in timesteps}
    return (result, min(result, key=result.get))


def refine_harmonized_generation(*, source, target, source_title, training_queries,
                                 anchor, initial, teacher, mask_path, source_foreground,
                                 parent_anchor, print_regions, output, surrogate, purifier,
                                 config=PerLAGELatentConfig(), pause_path=None, halo_reference=None):
    config.validate()
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    tick = time.perf_counter()
    device = surrogate.device
    g0 = read_tensor(anchor, device)
    start = read_tensor(initial, device)
    owned = read_tensor(teacher, device)
    original = read_tensor(source, device)
    oldmask_array = np.asarray(Image.open(relocate_io_path(mask_path)).convert('L')).copy()
    oldmask = torch.as_tensor(oldmask_array, device=device, dtype=torch.float32)[None, None] / 255
    fixed_pixel = start - g0
    if (fixed_pixel - pixel_projection(g0, fixed_pixel, oldmask, config.auxiliary_levels)).abs().max() > 1e-06:
        raise ValueError('原辅助残差越界')
    foreground = read_rgb(source_foreground)[..., 0] > 0
    roi = region_mask(read_rgb(source).shape, print_regions)
    macro_mask = oldmask_array.astype(float) / 255 * foreground * ~roi
    macro = torch.as_tensor(macro_mask, device=device, dtype=torch.float32)[None, None]
    identity = json.loads(json.dumps(dict(
        method='perlage_harmonization_latent_v1',
        config=asdict(config),
        inputs={k: sha256_file(v) for k, v in dict(
            source=source, target=target, anchor=anchor, initial=initial,
            teacher=teacher, mask=mask_path, source_foreground=source_foreground,
            parent_anchor=parent_anchor).items()},
        encoder=surrogate.snapshot(),
        purifier=purifier.snapshot(),
        code_sha256=sha256_file(__file__),
        no_pixel_optimization=True,
        original_auxiliary_residual_fixed=True,
        printed_region_fixed_to_harmonized_anchor=True,
        private_gallery_used=False,
        true_rank_used=False,
        full_gradient_original_latent_route=True
    )))
    identity.update(
        experimental_protocol='dominant_color_full_pipeline_v1',
        halo_reference_sha256=sha256_file(halo_reference) if halo_reference is not None else None,
        training_data_policy=TRAINING_DATA_POLICY,
        purchase_history_used=False,
        victim_memory_used=False,
        source_title=source_title,
        training_queries=list(training_queries),
        shadow_pool_used=False,
        selection_rule='unthresholded_owned_query_scores'
    )
    ip = root / 'identity.json'
    if ip.exists() and json.loads(ip.read_text(encoding='utf-8')) != identity:
        raise ValueError('生成细化身份改变')
    if not ip.exists() and any(root.iterdir()):
        raise ValueError('拒绝覆盖未知生成目录')
    write_json(ip, identity)
    if (root / 'metadata.json').exists():
        m = json.loads((root / 'metadata.json').read_text(encoding='utf-8'))
        if m.get('status') == 'completed':
            if sha256_file(root / 'adversarial.png') != m['output_sha256']:
                raise ValueError('完成生成图变化')
            return (root / 'adversarial.png', m)
    Image.fromarray(np.rint(macro_mask * 255).astype(np.uint8)).save(root / 'latent_edit_mask.png')
    surrogate.model.eval().requires_grad_(False)
    with torch.no_grad():
        target_feature = surrogate.encode_image(read_tensor(target, device)).global_embedding.detach()
    joint = JointRetrievalObjective(surrogate, target_feature, source_title, training_queries)
    write_json(root / 'public_title_encoding_audit.json', joint.audit)
    branch = SourceLatentBranch(g0, macro, purifier, config)
    optimizer = torch.optim.Adam([branch.delta], lr=config.latent_learning_rate)
    context = prepare_print_context(read_rgb(source), read_rgb(parent_anchor), oldmask_array, foreground, print_regions)
    parent_halo = halo_error(context, read_rgb(halo_reference if halo_reference is not None else teacher))
    teacher_features = {}
    with torch.no_grad():
        ordinary_teacher = surrogate.encode_image(transformed_batch(owned)).global_embedding.detach()
    ordinary_floors = joint.scores(ordinary_teacher) - config.ordinary_tolerance
    with torch.no_grad():
        for t in config.timesteps:
            for s in set(config.development_seeds) | set(config.selection_seeds):
                teacher_features[t, s] = surrogate.encode_image(
                    quantize_ste(purifier(owned, seed=s, start_timestep=t))).global_embedding.detach()

    def compose():
        generation = quantize_ste((g0 + branch.residual()).clamp(0, 1))
        image = quantize_ste((generation + pixel_projection(generation, fixed_pixel, oldmask, config.auxiliary_levels)).clamp(0, 1))
        return (image, generation)

    @torch.no_grad()
    def quality(image):
        ss = float(source_ssim(image, original))
        mae = float((image - original).abs().mean() * 255)
        return (ss, mae)

    @torch.no_grad()
    def development(image, step):
        ordinary = joint.scores(surrogate.encode_image(transformed_batch(image)).global_embedding)
        slack = float((ordinary - ordinary_floors).min())
        ss, mae = quality(image)
        gaps = []
        selection_scores = []
        distances = []
        for t in config.timesteps:
            for s in (*config.development_seeds, *config.selection_seeds):
                feature = surrogate.encode_image(quantize_ste(purifier(image, seed=s, start_timestep=t))).global_embedding
                scores = joint.scores(feature)
                ref = teacher_features[t, s]
                if s in config.development_seeds:
                    gap = scores - joint.scores(ref)
                    gaps.append(dict(timestep=t, seed=s, minimum_query_gap=float(gap.min()), query_gaps=gap.cpu().tolist()))
                    key = F.normalize(F.normalize(feature.float(), dim=-1) + joint.title_feature, dim=-1)
                    refkey = F.normalize(F.normalize(ref.float(), dim=-1) + joint.title_feature, dim=-1)
                    distances.append(float((key - refkey).norm(dim=-1).max()))
                else:
                    selection_scores.append(float(pool_free_value(scores, config.temperature)))
        group, weakest = relative_noise_panel(gaps, config.timesteps)
        retention = min(group.values()) + config.retention_tolerance
        arr = np.rint(image[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        halo = halo_error(context, arr)
        natural = halo <= max(parent_halo * 0.5, 0.5)
        good = ss >= config.source_ssim_floor and mae <= config.source_mae_ceiling
        vals = image.new_tensor(selection_scores)
        utility = float(0.5 * vals.mean() + 0.5 * softmin(vals, config.temperature))
        return dict(
            step=step,
            ordinary_guard_slack=slack,
            source_ssim=ss,
            source_mae_levels=mae,
            quality_ok=good,
            halo_error=halo,
            parent_halo_error=parent_halo,
            natural_halo_gate=natural,
            noise_panel=gaps,
            group_quartile_gaps=group,
            weakest_timestep=weakest,
            retention_slack=retention,
            max_own_key_distance=max(distances),
            selection_utility=utility,
            selection=[int(natural), int(good), int(slack >= -1e-06), min(slack + 1e-06, 0), int(retention >= 0), min(retention, 0), utility]
        )

    history = []
    best = None
    best_row = None
    step_done = 0
    previous = 0.0
    cp = root / 'checkpoint.pt'
    accepted = rejected = 0
    if cp.exists():
        state = torch.load(relocate_io_path(cp), map_location=device, weights_only=False)
        if state['identity'] != identity:
            raise ValueError('生成断点身份改变')
        with torch.no_grad():
            branch.delta.copy_(state['latent'])
        optimizer.load_state_dict(state['optimizer'])
        history = state['history']
        best = state['best']
        best_row = state['best_row']
        step_done = state['step']
        previous = state['seconds']
        accepted = state['accepted']
        rejected = state['rejected']

    def save(status):
        pending = cp.with_suffix('.tmp')
        torch.save(dict(
            identity=identity,
            latent=branch.delta.detach(),
            optimizer=optimizer.state_dict(),
            history=history,
            best=best,
            best_row=best_row,
            step=step_done,
            seconds=previous + time.perf_counter() - tick,
            accepted=accepted,
            rejected=rejected
        ), relocate_io_path(pending))
        pending.replace(cp)
        write_json(root / 'progress.json', dict(
            status=status,
            step=step_done,
            seconds=previous + time.perf_counter() - tick,
            best=best_row,
            accepted_updates=accepted,
            rejected_updates=rejected
        ))

    if best_row is None:
        best = branch.delta.detach().clone()
        best_row = development(compose()[0], 0)
        history.append(best_row)
        save('running')
        print(json.dumps(dict(step=0, selection=best_row['selection'], groups=best_row['group_quartile_gaps'])), flush=True)

    for step in range(step_done + 1, config.steps + 1):
        if pause_path and Path(pause_path).exists():
            save('paused')
            return (None, dict(status='paused'))
        image = compose()[0].detach().requires_grad_(True)
        normals = []
        slacks = []
        structure = source_ssim(image, original)
        normal, = torch.autograd.grad(structure, image)
        normals.append(normal.detach())
        slacks.append(float(structure.detach()) - config.source_ssim_floor)
        average = (image - original).abs().mean()
        normal, = torch.autograd.grad(-average, image)
        normals.append(normal.detach())
        slacks.append(config.source_mae_ceiling / 255 - float(average.detach()))
        features = surrogate.encode_image(transformed_batch(image)).global_embedding
        loss = owned_retention_loss(features, ordinary_teacher, joint, config)
        unit = (image - start) / (config.latent_levels / 255)
        loss += config.macro_l2_weight * unit.square().mean() + config.macro_tv_weight * (
            (unit[..., 1:, :] - unit[..., :-1, :]).abs().mean() + (unit[..., :, 1:] - unit[..., :, :-1]).abs().mean())
        gradient, = torch.autograd.grad(loss, image)
        timestep = int(history[-1]['weakest_timestep'])
        derivatives = []
        values = []
        for sample in range(config.full_eot_samples):
            seed = config.training_seed_base + step * 10 + sample
            with torch.no_grad():
                ref = surrogate.encode_image(quantize_ste(purifier(owned, seed=seed, start_timestep=timestep))).global_embedding.detach()
            feature = surrogate.encode_image(quantize_ste(purifier(image, seed=seed, start_timestep=timestep, gradient='full'))).global_embedding
            robust = owned_retention_loss(feature, ref, joint, config)
            derivative, = torch.autograd.grad(robust, image)
            derivatives.append(derivative)
            values.append(float(robust.detach()))
        weights = 0.5 / len(values) + 0.5 * torch.softmax(image.new_tensor(values) / config.temperature, 0)
        gradient += config.diffusion_weight * sum((w * g for w, g in zip(weights, derivatives)))
        if not torch.isfinite(gradient).all():
            raise FloatingPointError('完整图像梯度非有限')
        old = branch.delta.detach().clone()
        old_optimizer = copy.deepcopy(optimizer.state_dict())
        optimizer.zero_grad(set_to_none=True)
        decoded = compose()[0]
        latent_normals = []
        for normal in normals:
            derivative, = torch.autograd.grad(decoded, branch.delta, grad_outputs=normal, retain_graph=True)
            latent_normals.append(derivative.detach())
        decoded.backward(gradient)
        if branch.delta.grad is None or not torch.isfinite(branch.delta.grad).all():
            raise FloatingPointError('原潜变量梯度非有限')
        optimizer.param_groups[0]['lr'] = config.latent_learning_rate * (1 - 0.75 * step / config.steps)
        optimizer.step()
        with torch.no_grad():
            branch.delta.clamp_(-config.latent_limit, config.latent_limit)
            smooth = F.avg_pool2d(F.pad(branch.delta, (1, 1, 1, 1), mode='reflect'), 3, stride=1)
            branch.delta.lerp_(smooth, config.latent_smoothing_mix)
            branch.delta.copy_((old + tangent_projection(branch.delta - old, latent_normals, slacks)).clamp(-config.latent_limit, config.latent_limit))
            proposed = branch.delta.detach().clone()
            fraction = 0.0
            for backtrack in range(config.guard_backtracks + 1):
                scale = 0.5 ** backtrack
                branch.delta.copy_(old + scale * (proposed - old))
                candidate = compose()[0]
                ss, mae = quality(candidate)
                if ss < config.source_ssim_floor or mae > config.source_mae_ceiling:
                    continue
                ordinary = joint.scores(surrogate.encode_image(transformed_batch(candidate)).global_embedding)
                if not bool((ordinary >= ordinary_floors - 1e-07).all()):
                    continue
                arr = np.rint(candidate[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                if halo_error(context, arr) > max(parent_halo * 0.5, 0.5):
                    continue
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
            image = compose()[0].detach()
            row = development(image, step)
            history.append(row)
            save_tensor_image(root / f'checkpoint_{step:04d}.png', image)
            if tuple(row['selection']) > tuple(best_row['selection']):
                best = branch.delta.detach().clone()
                best_row = row
            save('running')
            print(json.dumps(dict(
                step=step, best_step=best_row['step'], groups=row['group_quartile_gaps'],
                selection=row['selection'], seconds=previous + time.perf_counter() - tick,
                accepted=accepted, rejected=rejected, ssim=row['source_ssim'], mae=row['source_mae_levels']
            )), flush=True)

    with torch.no_grad():
        branch.delta.copy_(best)
        image, generation = compose()
    save_tensor_image(root / 'generative_anchor.png', generation)
    save_tensor_image(root / 'adversarial.png', image)
    a = read_rgb(root / 'adversarial.png')
    g = read_rgb(root / 'generative_anchor.png')
    old_g = read_rgb(anchor)
    aux = np.abs(a.astype(int) - g)
    macro_change = np.abs(g.astype(int) - old_g)
    if aux.max() > 4 or macro_change.max() > config.latent_levels or np.any(g[roi] != old_g[roi]):
        raise AssertionError('生成预算、辅助预算或打印区锁定失败')
    meta = dict(
        shadow_pool_used=False,
        selection_rule="unthresholded_owned_query_scores",
        status='completed',
        output_sha256=sha256_file(root / 'adversarial.png'),
        anchor_sha256=sha256_file(root / 'generative_anchor.png'),
        elapsed_seconds=previous + time.perf_counter() - tick,
        history=history,
        best=best_row,
        auxiliary_max_levels=int(aux.max()),
        auxiliary_mae_levels=float(aux.mean()),
        macro_max_levels=int(macro_change.max()),
        macro_mae_levels=float(macro_change.mean()),
        config=asdict(config),
        accepted_updates=accepted,
        rejected_updates=rejected,
        no_pixel_optimization=True,
        private_gallery_used=False,
        true_rank_used=False,
        full_gradient_original_latent_route=True
    )
    write_json(root / 'metadata.json', meta)
    save('completed')
    return (root / 'adversarial.png', meta)
