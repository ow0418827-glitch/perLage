"""原始像素优化的标准模块入口：三个代理、局部最优传输与像素投影。"""
from dataclasses import asdict, dataclass, replace
import gc
import time

import torch

from src.paths import ProjectPath as Path, PROJECT
from src.visual.io import (load_image, pil_to_tensor, save_tensor_png, save_json,
                           sha256_file, decoded_image_audit)
from .config import resolve_optimization
from .objective import TargetCache, prepare_targets, joint_objective, evaluate_objective
from .pixel import pixel_update
from .surrogates import load_surrogates, resolve_device


@dataclass(frozen=True)
class AttackResult:
    image: object
    tensor: torch.Tensor
    decoded_tensor: torch.Tensor
    metadata: dict


def run_attack(source_image, target_image, target_text, project_config, *,
               method='legacy_pixel', steps=None, output_directory=None,
               surrogates=None, target_cache=None):
    """保留原方法每步的联合计算图、聚类、权重更新顺序与最终步选择。"""
    if method not in ('lucid', 'legacy_pixel'):
        raise ValueError('此入口仅支持原始像素优化')
    project_config.validate()
    count = project_config.attack.reference_steps if steps is None else steps
    if type(count) is not int or count < 1:
        raise ValueError('完整像素优化步数必须为正整数')
    config = replace(project_config.attack, reference_steps=count)
    device = resolve_device(project_config.device, allow_fallback=False)
    torch.manual_seed(project_config.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(project_config.seed)
    started = time.perf_counter()
    models = list(surrogates) if surrogates is not None else load_surrogates(
        project_config.surrogates, device=str(device), precision=project_config.attack_precision,
        allow_downloads=False, allow_device_fallback=False,
        require_true_local_tokens=project_config.runtime.require_true_local_tokens,
        hash_weights=project_config.runtime.hash_model_weights)
    if len(models) != 3 or any(torch.device(m.device) != device for m in models):
        raise ValueError('三个代理必须位于指定设备')
    source_pil, target_pil = load_image(source_image), load_image(target_image)
    source, target = pil_to_tensor(source_pil, device), pil_to_tensor(target_pil, device)
    cache = target_cache or TargetCache()
    targets = prepare_targets(models, target, target_text, config, cache=cache,
                              prepare_cluster_centers=False)
    weights = source.new_full((3,), 1 / 3)
    delta = torch.zeros_like(source)
    history = []
    loop_started = time.perf_counter()
    for step in range(1, count + 1):
        parameter = delta.detach().requires_grad_(True)
        objective = joint_objective((source + parameter).clamp(0, 1), models, targets,
                                    weights, config, use_cached_target_centers=False)
        gradient, = torch.autograd.grad(objective.loss, parameter)
        delta, update = pixel_update(delta, gradient, source,
                                    step_size=config.spatial_step_size, epsilon=config.epsilon)
        previous_weights = weights
        weights = torch.softmax(-objective.global_similarities.detach(), dim=0)
        if step == 1 or step == count or step % config.log_every == 0:
            row = dict(step=step, **objective.detached_record(),
                       linf_after_update=update['resulting_linf'],
                       weights_used_for_this_step=previous_weights.tolist(),
                       weights_for_next_step=weights.tolist())
            history.append(row)
            print(f'像素优化 {step}/{count}，目标值 {row["loss"]:.6f}', flush=True)
    final = (source + delta).clamp(0, 1).detach()
    final_objective = evaluate_objective(final, models, targets, weights, config,
                                         use_cached_target_centers=False).detached_record()
    root = Path(output_directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    image = save_tensor_png(root / 'adversarial.png', final)
    decoded_pil = load_image(image)
    decoded = pil_to_tensor(decoded_pil, device)
    audit = decoded_image_audit(source_pil, decoded_pil,
                               maximum_levels=project_config.acceptance.decoded_max_pixel_difference)
    if not audit['within_budget']:
        raise AssertionError('实际落盘图片超过像素预算')
    metadata = dict(schema_version=2, method='lucid', status='success', seed=project_config.seed,
        output_image=str(image), output_image_sha256=sha256_file(image),
        output_metadata=str(root / 'attack_metrics.json'),
        attack=dict(**asdict(config), steps_requested=count, steps_completed=count,
                    stop_reason='requested_steps_completed', selected_step=count,
                    gradient_execution='joint_three_surrogate_graph', target_cluster_centers_cached=False),
        history=history, final_objective=final_objective,
        final_model_weights=weights.tolist(), decoded_image_audit=audit,
        surrogates=[m.snapshot() for m in models],
        timing=dict(optimization_loop_seconds=time.perf_counter() - loop_started,
                    total_seconds=time.perf_counter() - started),
        primary_evaluator_used_during_attack=False, victim_memory_used=False,
        formal_queries_used=False, private_gallery_used=False)
    save_json(root / 'attack_metrics.json', metadata)
    return AttackResult(decoded_pil, final.cpu(), decoded.cpu(), metadata)


def generate_attack(source, target, output_dir, *, optimization_config=None, steps=1000,
                    epsilon_levels=16, device=None, target_text=None, text_weight=1.0):
    if type(steps) is not int or steps < 1 or not 0 < epsilon_levels <= 255:
        raise ValueError('步数及像素预算无效')
    config = resolve_optimization(optimization_config)
    config = replace(config, device=device or config.device,
        attack=replace(config.attack, epsilon=epsilon_levels / 255,
                       text_weight=float(text_weight) if target_text is not None else 0.0),
        acceptance=replace(config.acceptance, decoded_max_pixel_difference=epsilon_levels))
    if target_text is not None:
        from src.visual.title_alignment import run_title_aligned_attack
        return run_title_aligned_attack(source, target, output_dir, config,
            target_text=target_text, steps=steps, run_attack=run_attack)
    if text_weight != 1.0:
        raise ValueError('文字权重需要目标商品名称')
    result = run_attack(source, target, None, config, steps=steps, output_directory=output_dir)
    output, metadata = Path(result.metadata['output_image']), result.metadata
    del result
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output, metadata
