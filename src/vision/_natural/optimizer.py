"""完整方法的内部初始化优化；本阶段完成不能替代最终潜变量细化。"""
from src.paths import relocate_io_path
from dataclasses import asdict, dataclass
from src.paths import ProjectPath as Path
import copy, hashlib, json, time
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from src.vision._natural.core import NaturalInitializationConfig, source_ssim, pixel_projection, public_development_value
from src.vision.image_utils import read_tensor, quantize_ste, save_tensor_image, sha256_file, write_json
from src.vision.retrieval_objective import JointRetrievalObjective, TRAINING_DATA_POLICY
from src.vision.robust_objectives import softmin
from src.vision.robust_objectives import transformed_batch
from src.vision.latent_branch import SourceLatentBranch
from src.vision.robust_objectives import compensated_targets
from src.vision.public_pool import margin_loss, PublicMarginConfig

@dataclass(frozen=True)
class InitializationOptimizationConfig(NaturalInitializationConfig):
    shadow_weight: float = 0.0
    ray_repair_iterations: int = 8

    def validate(self):
        super().validate()
        if isinstance(self.ray_repair_iterations, bool) or not isinstance(self.ray_repair_iterations, int) or (not 1 <= self.ray_repair_iterations <= 16):
            raise ValueError('射线修复检查次数必须在一至十六之间')

def repair_on_reference_ray(check, iterations):
    """只返回实际检查可行的点；非凸约束下不冒称全局最优投影。"""
    if check(1.0):
        return 1.0
    if not check(0.0):
        raise ValueError('阶段参考点不可行，拒绝射线修复')
    lower, upper = (0.0, 1.0)
    for _ in range(iterations):
        middle = (lower + upper) / 2
        if check(middle):
            lower = middle
        else:
            upper = middle
    if not check(lower):
        if not check(0.0):
            raise ValueError('阶段参考点复检失败')
        return 0.0
    return lower

def refine_generation_initialization(source, target, anchor, mask_path, output, *, surrogate, source_title, training_queries, purifier, public_bank=None, config=InitializationOptimizationConfig(), pause_path=None):
    if public_bank is not None or config.shadow_weight != 0:
        raise ValueError('当前完整方法的初始化禁止竞争池或记忆池输入')
    """新增独立入口；实际生成锚点由调用者提供，生成成本另行计入。"""
    config.validate()
    if config.shadow_weight and public_bank is None:
        raise ValueError('启用公开竞争目标需要可审计的公开池')
    output = Path(output)
    device = surrogate.device
    original, initial = (read_tensor(source, device), read_tensor(anchor, device))
    with Image.open(relocate_io_path(mask_path)) as im:
        mask = torch.as_tensor(np.array(im.convert('L')), device=device, dtype=torch.float32)[None, None] / 255
    if initial.shape != original.shape or mask.shape[-2:] != initial.shape[-2:]:
        raise ValueError('图片或保护区域大小不一致')
    if ((initial - original) * (mask == 0)).abs().max() != 0:
        raise ValueError('生成锚点已经改变受保护源像素')
    surrogate.model.eval().requires_grad_(False)
    with torch.no_grad():
        target_feature = surrogate.encode_image(read_tensor(target, device)).global_embedding.detach()
    joint = JointRetrievalObjective(surrogate, target_feature, source_title, training_queries)
    teachers, valid = compensated_targets(joint.query_features, joint.title_feature)
    identity = json.loads(json.dumps(dict(method='perlage_generation_initialization', config=asdict(config), inputs={k: sha256_file(p) for k, p in [('source', source), ('target', target), ('anchor', anchor), ('mask', mask_path)]}, source_title_sha256=hashlib.sha256(source_title.encode('utf-8')).hexdigest(), training_queries=list(training_queries), encoder=surrogate.snapshot(), purifier=purifier.snapshot(), public_bank_sha256=public_bank['sha256'] if public_bank is not None else None, code_sha256=sha256_file(__file__), private_gallery_used=False, true_rank_used=False, pixel_budget_reference='frozen_refined_generative_anchor', source_linf_budget_enforced=False, sequence='instruction_edit_then_latent_only_then_frozen_generation_plus_small_pixel', natural_manifold_certified=False)))
    identity.update(training_data_policy=TRAINING_DATA_POLICY, purchase_history_used=False, victim_memory_used=False,
                    objective=joint.audit)
    output.mkdir(parents=True, exist_ok=True)
    ip = output / 'identity.json'
    if ip.exists() and json.loads(ip.read_text(encoding='utf-8')) != identity:
        raise ValueError('方法身份改变，请另建版本')
    elif not ip.exists() and any(output.iterdir()):
        raise ValueError('拒绝覆盖未知实验目录')
    write_json(ip, identity)
    if (output / 'metadata.json').exists():
        meta = json.loads((output / 'metadata.json').read_text(encoding='utf-8'))
        if meta['status'] == 'completed':
            if sha256_file(output / 'adversarial.png') != meta['output_sha256']:
                raise ValueError('完成图片改变')
            return (output / 'adversarial.png', meta)
    started, previous = (time.perf_counter(), 0.0)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    branch = SourceLatentBranch(initial, mask, purifier, config)
    optimizer = torch.optim.Adam([branch.delta], lr=config.latent_learning_rate)
    pixel, velocity = (torch.zeros_like(original), torch.zeros_like(original))
    generative = initial.clone()
    history, artifacts = ([], {})
    cp = output / 'checkpoint.pt'
    phases = [('latent_clean', config.latent_clean_steps), ('latent_diffusion', config.latent_diffusion_steps), ('pixel_clean', config.pixel_clean_steps), ('pixel_diffusion', config.pixel_diffusion_steps)]
    phase_index, phase_step, floors = (0, 0, None)
    best, best_row, latest = (initial.clone(), None, None)
    best_latent, best_pixel = (branch.delta.detach().clone(), pixel.clone())
    accepted, rejected, ray_repairs = (0, 0, 0)
    reference_latent, reference_pixel = (branch.delta.detach().clone(), pixel.clone())

    def compose():
        if phase_index < 2:
            return quantize_ste((initial + branch.residual()).clamp(0, 1))
        return quantize_ste((generative + pixel_projection(generative, pixel, mask, config.pixel_levels)).clamp(0, 1))

    def scores(image):
        return joint.scores(surrogate.encode_image(quantize_ste(image)).global_embedding)

    def utility(image, thresholds=None):
        features = surrogate.encode_image(quantize_ste(image)).global_embedding
        semantic = phase_index < 2 or robust
        joint_weight = config.semantic_joint_weight if semantic else 1.0
        teacher_weight = config.semantic_teacher_weight if semantic else config.teacher_weight
        image_weight = config.semantic_image_weight if semantic else config.image_weight
        value = joint_weight * torch.stack([softmin(q, config.temperature) for q in joint.scores(features)])
        if valid.any():
            value = value + teacher_weight * torch.stack([softmin(q, config.temperature) for q in features @ teachers[valid].T])
        value = value + image_weight * F.cosine_similarity(features, target_feature)
        if thresholds is not None and config.shadow_weight:
            value = value - config.shadow_weight * margin_loss(features, joint, target_feature, thresholds, PublicMarginConfig())
        return value

    def penalty(image):
        reference = initial if phase_index < 2 else generative
        levels = config.latent_levels if phase_index < 2 else max(config.pixel_levels, 1)
        unit = (image - reference) / (levels / 255)
        tv = (unit[..., 1:, :] - unit[..., :-1, :]).abs().mean() + (unit[..., :, 1:] - unit[..., :, :-1]).abs().mean()
        return config.tv_weight * tv + config.l2_weight * unit.square().mean()

    def active_ssim_floor():
        return max(config.source_ssim_floor, config.ordinary_ssim_floor) if phase_index < 3 else config.source_ssim_floor

    @torch.no_grad()
    def quality_ok(image):
        return float(source_ssim(image, original)) >= active_ssim_floor() + config.ssim_safety_buffer and float((image - original).abs().mean() * 255) <= config.source_mae_ceiling + 1e-05

    @torch.no_grad()
    def validate(image, step, robust):
        ordinary = scores(transformed_batch(image))
        ss, mae = (float(source_ssim(image, original)), float((image - original).abs().mean() * 255))
        row = dict(phase=phases[phase_index][0], step=step, ordinary_scores=ordinary.cpu().tolist(), source_ssim=ss, source_mae=mae, active_ssim_floor=active_ssim_floor(), quality_ok=quality_ok(image), diffusion=[])
        guard = 0.0 if floors is None else float((ordinary - floors).min())
        row['guard_min'] = guard
        if robust:
            for timestep in config.diffusion_timesteps:
                for seed in config.validation_seeds:
                    out = quantize_ste(purifier(image, seed=seed, start_timestep=timestep))
                    current_scores = scores(out)
                    item = dict(timestep=timestep, seed=seed, utility=float(softmin(current_scores.flatten(), config.temperature)))
                    if public_bank is not None:
                        threshold = public_bank['development_diffusion'][f'{timestep}_{seed}']['joint3']
                        gap = current_scores - threshold
                        item.update(public_margin_min=float(gap.min()), public_margin_value=float(public_development_value(current_scores, threshold, temperature=config.development_temperature)))
                    else:
                        item['public_margin_value'] = item['utility']
                    row['diffusion'].append(item)
            v = image.new_tensor([r['public_margin_value'] for r in row['diffusion']])
            value = float(0.5 * v.mean() + 0.5 * softmin(v, config.development_temperature))
        else:
            value = float(softmin(ordinary.flatten(), config.temperature))
        row['selection'] = [int(row['quality_ok']), int(guard >= -1e-06), min(guard + 1e-06, 0), value - float(penalty(image))]
        return row
    if cp.exists():
        state = torch.load(relocate_io_path(cp), map_location=device, weights_only=False)
        if state['identity'] != identity:
            raise ValueError('断点身份改变')
        with torch.no_grad():
            branch.delta.copy_(state['latent'])
        optimizer.load_state_dict(state['optimizer'])
        for name in ('pixel', 'velocity', 'generative', 'history', 'artifacts', 'phase_index', 'phase_step', 'floors', 'best', 'best_row', 'latest', 'accepted', 'rejected'):
            if name not in state:
                raise ValueError('断点缺少必要状态')
        pixel, velocity, generative = (state['pixel'], state['velocity'], state['generative'])
        history, artifacts = (state['history'], state['artifacts'])
        phase_index, phase_step, floors = (state['phase_index'], state['phase_step'], state['floors'])
        best, best_row, latest = (state['best'], state['best_row'], state['latest'])
        best_latent, best_pixel = (state['best_latent'], state['best_pixel'])
        accepted, rejected, previous = (state['accepted'], state['rejected'], state['seconds'])
        ray_repairs = state['ray_repairs']
        reference_latent = state['reference_latent']
        reference_pixel = state['reference_pixel']

    def save(status):
        pending = cp.with_suffix('.tmp')
        torch.save(dict(identity=identity, latent=branch.delta.detach(), optimizer=optimizer.state_dict(), pixel=pixel.detach(), velocity=velocity.detach(), generative=generative.detach(), history=history, artifacts=artifacts, phase_index=phase_index, phase_step=phase_step, floors=floors, best=best.detach(), best_row=best_row, latest=latest, accepted=accepted, rejected=rejected, best_latent=best_latent, best_pixel=best_pixel, reference_latent=reference_latent, reference_pixel=reference_pixel, ray_repairs=ray_repairs, seconds=previous + time.perf_counter() - started), relocate_io_path(pending))
        pending.replace(cp)
        write_json(output / 'progress.json', dict(status=status, phase=phases[phase_index][0] if phase_index < 4 else 'finished', phase_step=phase_step, seconds=previous + time.perf_counter() - started, latest=latest, accepted_updates=accepted, rejected_updates=rejected, ray_repairs=ray_repairs))
    while phase_index < len(phases):
        name, count = phases[phase_index]
        robust = name.endswith('diffusion')
        if best_row is None:
            best = compose().detach()
            reference_latent = branch.delta.detach().clone()
            reference_pixel = pixel.clone()
            if robust:
                with torch.no_grad():
                    tolerance = config.latent_guard_tolerance if phase_index < 2 else config.guard_tolerance
                    floors = scores(transformed_batch(best)).detach() - tolerance
            else:
                floors = None
            best_row = latest = validate(best, 0, robust)
            history.append(best_row)
            best_latent = branch.delta.detach().clone()
            best_pixel = pixel.clone()
        for step in range(phase_step + 1, count + 1):
            if pause_path and Path(pause_path).exists():
                save('paused')
                return (output / 'adversarial.png', dict(status='paused'))
            candidate = compose().detach().requires_grad_(True)
            thresholds = public_bank['train_ordinary'] if public_bank is not None else None
            if robust:
                ordinary_scores = scores(transformed_batch(candidate))
                objective = -0.008 * F.softplus((floors - ordinary_scores) / 0.008).mean()
            else:
                objective = softmin(utility(transformed_batch(candidate), thresholds), config.temperature)
            objective = objective - penalty(candidate)
            gradient, = torch.autograd.grad(objective, candidate)
            timestep = None
            if robust:
                timestep = config.diffusion_timesteps[(step - 1) % len(config.diffusion_timesteps)]
                values, gradients = ([], [])
                for sample in range(config.full_eot_samples):
                    seed = config.training_seed_base + phase_index * 100000 + step * 10 + sample
                    purified = purifier(candidate, seed=seed, start_timestep=timestep, gradient='full')
                    thresholds = public_bank['train_diffusion'][str(timestep)] if public_bank is not None else None
                    value = utility(purified, thresholds).mean()
                    dg, = torch.autograd.grad(value, candidate)
                    values.append(float(value.detach()))
                    gradients.append(dg)
                weights = 0.5 / len(values) + 0.5 * torch.softmax(-candidate.new_tensor(values) / 0.02, 0)
                gradient = gradient + config.diffusion_weight * sum((w * g for w, g in zip(weights, gradients)))
            if not torch.isfinite(gradient).all():
                raise FloatingPointError('梯度不是有限数')
            old_latent, old_pixel, old_velocity = (branch.delta.detach().clone(), pixel.clone(), velocity.clone())
            old_optimizer = copy.deepcopy(optimizer.state_dict())
            if phase_index < 2:
                optimizer.zero_grad(set_to_none=True)
                decoded = compose()
                decoded.backward(-gradient)
                optimizer.param_groups[0]['lr'] = config.latent_learning_rate * (1 - 0.75 * step / max(count, 1))
                optimizer.step()
                with torch.no_grad():
                    branch.delta.clamp_(-config.latent_limit, config.latent_limit)
                    smooth = F.avg_pool2d(F.pad(branch.delta, (1, 1, 1, 1), mode='reflect'), 3, stride=1)
                    branch.delta.lerp_(smooth, config.latent_smoothing_mix)
            else:
                with torch.no_grad():
                    velocity = config.momentum * velocity + gradient / gradient.abs().mean().clamp_min(1e-12)
                    lr = (config.pixel_start_levels + (config.pixel_end_levels - config.pixel_start_levels) * step / max(count, 1)) / 255
                    proposed = pixel_projection(generative, pixel + lr * velocity.sign(), mask, config.pixel_levels).detach()
                    pixel = proposed
            proposed_latent, proposed_pixel = (branch.delta.detach().clone(), pixel.clone())
            with torch.no_grad():

                def feasible(fraction):
                    nonlocal pixel
                    branch.delta.copy_(reference_latent + fraction * (proposed_latent - reference_latent))
                    pixel = reference_pixel + fraction * (proposed_pixel - reference_pixel)
                    image = compose()
                    if not quality_ok(image):
                        return False
                    return floors is None or bool((scores(transformed_batch(image)) >= floors - 1e-07).all())
                fraction = repair_on_reference_ray(feasible, config.ray_repair_iterations)
                if fraction < 1:
                    ray_repairs += 1
                if fraction:
                    accepted += 1
                    velocity = velocity.detach()
                else:
                    rejected += 1
                    branch.delta.copy_(old_latent)
                    pixel, velocity = (old_pixel, old_velocity)
                    optimizer.load_state_dict(old_optimizer)
            phase_step = step
            if step % config.check_every == 0 or step == count:
                latest = validate(compose().detach(), step, robust)
                history.append(latest)
                if tuple(latest['selection']) > tuple(best_row['selection']):
                    best, best_row = (compose().detach(), latest)
                    best_latent, best_pixel = (branch.delta.detach().clone(), pixel.clone())
                save('running')
                print(json.dumps(dict(phase=name, step=step, best_step=best_row['step'], timestep=timestep, ssim=latest['source_ssim'], mae=latest['source_mae'], selection=latest['selection'], seconds=previous + time.perf_counter() - started)), flush=True)
        with torch.no_grad():
            branch.delta.copy_(best_latent)
            pixel = best_pixel
            result = compose().detach()
        save_tensor_image(output / (name + '.png'), result)
        artifacts[name] = dict(image=str((output / (name + '.png')).resolve()), sha256=sha256_file(output / (name + '.png')), best=best_row)
        if phase_index == 1:
            generative = result.clone()
            save_tensor_image(output / 'generative_anchor.png', generative)
        phase_index += 1
        phase_step = 0
        best_row = latest = None
        floors = None
        save('running')
    result = quantize_ste((generative + pixel_projection(generative, pixel, mask, config.pixel_levels)).clamp(0, 1)).detach()
    save_tensor_image(output / 'adversarial.png', result)
    saved = read_tensor(output / 'adversarial.png', device)
    relative = (saved - generative).abs() * 255
    if relative.max() > config.pixel_levels + 0.0001 or ((saved - original) * (mask == 0)).abs().max() != 0:
        raise AssertionError('实际保存图片违反像素收尾预算或保护区')
    if not quality_ok(saved):
        raise AssertionError('实际保存图片违反预先固定的外观门槛')
    meta = dict(status='completed', method='perlage_generation_initialization', config=asdict(config), history=history, artifacts=artifacts, output_sha256=sha256_file(output / 'adversarial.png'), generative_anchor_sha256=sha256_file(output / 'generative_anchor.png'), elapsed_seconds=previous + time.perf_counter() - started, initial_generation_cost_included=False, source_ssim=float(source_ssim(saved, original)), source_mae_levels=float((saved - original).abs().mean() * 255), source_max_levels=float((saved - original).abs().max() * 255), pixel_residual_max_levels=float(relative.max()), pixel_residual_mae_levels=float(relative.mean()), accepted_updates=accepted, rejected_updates=rejected, ray_repairs=ray_repairs, feasibility_repair='fixed_phase_reference_ray_actual_validation', peak_allocated_bytes=torch.cuda.max_memory_allocated() if device.type == 'cuda' else 0, private_gallery_used=False, true_rank_used=False)
    write_json(output / 'metadata.json', meta)
    save('completed')
    return (output / 'adversarial.png', meta)
