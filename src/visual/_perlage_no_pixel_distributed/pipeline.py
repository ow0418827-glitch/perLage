"""分散主体与背景变化的独立潜变量实验，按公开目标边际增益自适应结束。

只复用同组已完成且摘要通过的前三阶段，不读取共享库或正式查询。
前景和背景分别设定解码变化预算、原图质量门槛及归一化外观惩罚。
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
from scipy.ndimage import distance_transform_edt
import torch
import torch.nn.functional as F

from src.paths import PROJECT, ProjectPath as Path, relocate_io_path
from src.visual.image_utils import read_tensor, quantize_ste, save_tensor_image, sha256_file, write_json
from src.visual._perlage.components.colour import read_rgb, print_region_mask
from src.visual._perlage.components.core import source_ssim, tangent_projection
from src.visual._perlage.components.evidence import STAGES, stage_record
from src.visual._perlage_no_pixel_rankfocus.pipeline import (
    _VaeOnly, PublicFullKeyObjective, build_source_public_key, validate_prefix,
)
from src.visual.retrieval_objective import TRAINING_DATA_POLICY, training_data
from src.visual.robust_objectives import softmin


METHOD = 'PerLAGE_no_pixel_distributed_experiment'
DEFINITION = 'public_fullkey_regional_latent_budgets_adaptive_plateau_v1'


@dataclass(frozen=True)
class DistributedLatentConfig:
    steps: int = 1000
    check_every: int = 20
    min_steps: int = 300
    early_window_steps: int = 100
    early_min_gain: float = 0.004
    early_patience_windows: int = 2
    foreground_levels: int = 24
    background_levels: int = 12
    latent_working_size: int = 512
    latent_limit: float = 2.0
    latent_learning_rate: float = 0.03
    latent_smoothing_mix: float = 0.03
    source_ssim_floor: float = 0.85
    foreground_ssim_floor: float = 0.85
    foreground_mae_ceiling: float = 10.0
    background_mae_ceiling: float = 4.0
    background_added_mae_ceiling: float = 4.0
    deviation_l2_weight: float = 0.001
    residual_tv_weight: float = 0.0005
    texture_weight: float = 0.0005
    edge_feather_pixels: float = 2.0
    guard_backtracks: int = 8
    temperature: float = 0.04
    auxiliary_levels: int = 0
    feature_weight: float = 0.0
    diffusion_weight: float = 0.0
    seed: int = 42

    def validate(self):
        integers = ('steps', 'check_every', 'min_steps', 'early_window_steps',
                    'early_patience_windows', 'foreground_levels', 'background_levels',
                    'latent_working_size', 'guard_backtracks', 'seed')
        if any(type(getattr(self, k)) is not int for k in integers):
            raise ValueError('区域预算与停止日程必须使用整数')
        if (self.steps != 1000 or self.min_steps != 300 or self.check_every != 20
                or self.early_window_steps != 100 or self.early_patience_windows != 2
                or self.early_min_gain != 0.004):
            raise ValueError('本轮须遵守冻结的公开边际停止日程')
        if not (1 <= self.foreground_levels <= 64 and 0 <= self.background_levels <= 32):
            raise ValueError('前景或背景变化预算无效')
        if not (0.75 <= self.foreground_ssim_floor <= 0.99 and 0.75 <= self.source_ssim_floor <= 0.99):
            raise ValueError('区域结构相似度门槛无效')
        if not (0 < self.foreground_mae_ceiling <= 20 and 0 < self.background_mae_ceiling <= 8
                and 0 < self.background_added_mae_ceiling <= 8):
            raise ValueError('区域平均变化门槛无效')
        if self.latent_working_size != 512 or self.seed != 42:
            raise ValueError('本轮固定自编码器尺寸和种子')
        if not (0 < self.latent_limit <= 3 and 0 < self.latent_learning_rate <= .05
                and 0 <= self.latent_smoothing_mix <= .1 and 0 <= self.guard_backtracks <= 12):
            raise ValueError('潜变量优化参数无效')
        if self.auxiliary_levels != 0 or self.feature_weight != 0 or self.diffusion_weight != 0:
            raise ValueError('本轮禁止像素残差、原图检索特征保持及净化目标')
        if (min(self.deviation_l2_weight, self.residual_tv_weight, self.texture_weight) <= 0
                or self.edge_feather_pixels != 2 or not 0 < self.temperature <= 1):
            raise ValueError('本轮必须保留按区域归一化的外观约束')
        if any(not math.isfinite(v) for v in asdict(self).values() if isinstance(v, float)):
            raise ValueError('参数必须有限')


DISTRIBUTED_PRESETS = {
    'foreground_control': asdict(DistributedLatentConfig(background_levels=0)),
    'distributed_balanced': asdict(DistributedLatentConfig()),
    'distributed_flexible': asdict(DistributedLatentConfig(
        foreground_levels=48, background_levels=24, source_ssim_floor=.80,
        foreground_ssim_floor=.80, foreground_mae_ceiling=16.0,
        background_mae_ceiling=6.0, background_added_mae_ceiling=6.0)),
}
PRESETS = DISTRIBUTED_PRESETS


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _canonical(value):
    return json.loads(json.dumps(value, sort_keys=True, ensure_ascii=False))


def _files(paths):
    result = {}
    for p in paths:
        path = Path(p).resolve()
        value = sha256_file(path)
        if not value:
            raise FileNotFoundError('绑定文件缺失：' + str(path))
        result[str(path)] = value
    return result


def build_budget_map(foreground, print_regions_mask, config):
    """返回每个像素相对冻结起点的变化级数，印刷区及其边缘精确归零。"""
    foreground, protected = np.asarray(foreground, bool), np.asarray(print_regions_mask, bool)
    if foreground.ndim != 2 or foreground.shape != protected.shape or not foreground.any() or foreground.all():
        raise ValueError('预算图需要非空、非整图前景及同尺寸保护区')
    inside = np.clip(distance_transform_edt(foreground) / config.edge_feather_pixels, 0, 1)
    outside = np.clip(distance_transform_edt(~foreground) / config.edge_feather_pixels, 0, 1)
    print_feather = (np.clip(distance_transform_edt(~protected) / config.edge_feather_pixels, 0, 1)
                     if protected.any() else np.ones_like(foreground, dtype=float))
    return (config.foreground_levels * foreground * inside
            + config.background_levels * ~foreground * outside) * print_feather * ~protected


def _mask_tensor(mask, image):
    if isinstance(mask, torch.Tensor):
        value = mask.to(device=image.device, dtype=image.dtype)
    else:
        value = torch.as_tensor(np.asarray(mask).copy(), device=image.device, dtype=image.dtype)
    if value.ndim == 2:
        value = value[None, None]
    if value.shape != (1, 1, *image.shape[-2:]):
        raise ValueError('区域掩码形状不一致')
    return value


def _region_mean(value, mask):
    denominator = mask.sum() * value.shape[1]
    if float(denominator.detach()) <= 0:
        raise ValueError('区域统计不能使用空掩码')
    return (value * mask).sum() / denominator


def _ssim_map(image, source):
    image, source = image.double(), source.double()
    ux, uy = F.avg_pool2d(image, 7, stride=1), F.avg_pool2d(source, 7, stride=1)
    vx = (F.avg_pool2d(image.square(), 7, stride=1) - ux.square()) * 49 / 48
    vy = (F.avg_pool2d(source.square(), 7, stride=1) - uy.square()) * 49 / 48
    vxy = (F.avg_pool2d(image * source, 7, stride=1) - ux * uy) * 49 / 48
    return ((2 * ux * uy + .01 ** 2) * (2 * vxy + .03 ** 2)
            / ((ux.square() + uy.square() + .01 ** 2) * (vx + vy + .03 ** 2)))


def region_quality(image, original, anchor, foreground, config):
    """返回可微统计；区域均值各自归一化，不让背景稀释前景误差。"""
    fg = _mask_tensor(foreground, image)
    bg = 1 - fg
    difference, added = (image - original).abs() * 255, (image - anchor).abs() * 255
    structural = _ssim_map(image, original)
    fg_windows = F.avg_pool2d(fg.double(), 7, stride=1)
    bg_windows = 1 - fg_windows
    return dict(source_ssim=structural.mean(),
        source_mae_levels=difference.mean(),
        foreground_ssim=_region_mean(structural, fg_windows),
        background_ssim=_region_mean(structural, bg_windows),
        foreground_mae_levels=_region_mean(difference, fg),
        background_mae_levels=_region_mean(difference, bg),
        foreground_added_mae_levels=_region_mean(added, fg),
        background_added_mae_levels=_region_mean(added, bg))


def _quality_ok(q, config):
    return (q['source_ssim'] >= config.source_ssim_floor
            and q['foreground_ssim'] >= config.foreground_ssim_floor
            and q['foreground_mae_levels'] <= config.foreground_mae_ceiling
            and q['background_mae_levels'] <= config.background_mae_ceiling
            and q['background_added_mae_levels'] <= config.background_added_mae_ceiling)


def update_early_stopping(history, config):
    """由检查点历史完整重算停止判据，续跑不依赖易漂移的计数器。"""
    if not history or history[0].get('step') != 0:
        raise ValueError('停止判据需要从零步开始的完整训练检查点')
    steps = [r.get('step') for r in history]
    if steps != list(range(0, steps[-1] + 1, config.check_every)):
        raise ValueError('停止判据检查点缺失、重复或次序错误')
    if steps[-1] > config.steps or any(not math.isfinite(r['selection_utility']) for r in history):
        raise ValueError('停止日程或目标值无效')
    values, best = {}, -float('inf')
    for row in history:
        best = max(best, float(row['selection_utility']))
        values[row['step']] = best
    windows, stale, first_stop = [], 0, None
    for step in range(config.early_window_steps, steps[-1] + 1, config.early_window_steps):
        gain = values[step] - values[step - config.early_window_steps]
        eligible = step >= config.min_steps
        stale = stale + 1 if eligible and gain <= config.early_min_gain else 0
        windows.append(dict(step=step, reference_step=step-config.early_window_steps,
            best_public_utility=values[step], gain=gain, eligible=eligible, stale_windows=stale))
        if eligible and stale >= config.early_patience_windows and first_stop is None:
            first_stop = step
    completed = steps[-1]
    if first_stop is not None and completed > first_stop:
        raise ValueError('历史越过首次满足预注册停止规则的位置')
    reason = 'max_steps' if completed == config.steps else ('plateau' if first_stop == completed else 'running')
    return dict(rule='adaptive_public_plateau_v1', reason=reason,
        rule_satisfied=reason != 'running', steps_completed=completed, max_steps=config.steps,
        min_steps=config.min_steps, window_steps=config.early_window_steps,
        min_gain=config.early_min_gain, patience_windows=config.early_patience_windows,
        monitored_value='best_public_training_utility_minus_regional_visual_penalties', windows=windows,
        formal_rankings_used=False)


class _SpatialLatentBranch:
    def __init__(self, source, budgets, vae, config):
        from src.visual.diffusion_purification import working_shape
        self.source, self.budgets, self.vae = source, budgets / 255, vae
        with torch.no_grad():
            work = F.interpolate(source, size=working_shape(source.shape, config.latent_working_size),
                mode='bicubic', align_corners=False, antialias=True).clamp(0, 1)
            self.base = (vae.vae.encode(work * 2 - 1).latent_dist.mode().float()
                         * vae.vae.config.scaling_factor).detach()
            self.reference = vae._decode(self.base, source.shape).detach()
        self.delta = torch.zeros_like(self.base, requires_grad=True)

    def residual(self):
        decoded = self.vae._decode(self.base + self.delta, self.source.shape)
        return self.budgets * torch.tanh((decoded - self.reference) / self.budgets.clamp_min(1/255))


def validate_distributed_complete(ready, image=None):
    if (ready.get('method') != METHOD or ready.get('definition') != DEFINITION
            or ready.get('status') != 'completed'):
        raise ValueError('独立区域分布实验未完整完成')
    if ready.get('pixel_optimization') is not False or ready.get('auxiliary_max_levels') != 0:
        raise ValueError('区域分布实验出现像素优化或残差')
    for name in ('victim_memory_used', 'formal_queries_used', 'shadow_pool_used', 'true_rank_used', 'purchase_history_used'):
        if ready.get(name) is not False:
            raise ValueError('区域分布实验训练边界不完整：' + name)
    if ready.get('first_three_stages_reused') is not True or ready.get('fresh_first_three_stages_executed') is not False:
        raise ValueError('未如实记录前三阶段复用')
    stages = ready.get('stages', {})
    if set(stages) != set(STAGES):
        raise ValueError('缺少完整四阶段证据')
    for stage in stages.values():
        if stage.get('status') != 'completed' or not stage.get('files'):
            raise ValueError('四阶段证据不完整')
        for p, expected in stage['files'].items():
            if not expected or sha256_file(p) != expected:
                raise ValueError('阶段证据改变：' + p)
    identity_path = ready['method_identity']
    if sha256_file(identity_path) != ready.get('method_identity_sha256'):
        raise ValueError('方法身份摘要改变')
    identity = _read(identity_path)
    config = DistributedLatentConfig(**identity['config'])
    config.validate()
    if (identity.get('method') != METHOD or identity.get('definition') != DEFINITION
            or identity.get('profile') != ready.get('profile')
            or ready.get('profile') not in DISTRIBUTED_PRESETS
            or identity['config'] != DISTRIBUTED_PRESETS[ready['profile']]
            or ready.get('latent_config') != identity['config']):
        raise ValueError('方法档位配置绑定不符')
    data = training_data(identity['public_definition'])
    key = build_source_public_key(identity['source_public_record'], data['source_title'])
    if key != ready.get('source_public_key_text') or key != identity.get('source_public_key_text'):
        raise ValueError('源商品公开全文键改变')
    for role in ('source', 'target'):
        expected = identity['inputs'].get(str(Path(data[role]).resolve()))
        if not expected or sha256_file(data[role]) != expected or ready.get(role+'_sha256') != expected:
            raise ValueError('公开图对改变')
    if ready.get('prefix_binding') != identity['prefix_binding']:
        raise ValueError('复用前缀绑定改变')
    for p, expected in identity['prefix_binding']['files'].items():
        if sha256_file(p) != expected:
            raise ValueError('复用前缀证据改变')
    meta = _read(ready['metadata'])
    stop = update_early_stopping(meta['history'], config)
    last = stages[STAGES[-1]]
    if (stop != ready.get('early_stop') or stop != meta.get('early_stop') or not stop['rule_satisfied']
            or last.get('steps_completed') != stop['steps_completed']
            or last.get('steps_requested') != config.steps
            or last.get('schedule') != 'adaptive_public_plateau_v1'):
        raise ValueError('真实停止步数与预注册边际判据不符')
    if not _quality_ok(meta['final_check'], config):
        raise ValueError('最终图区域质量门槛不符')
    final = ready['candidates']['final']
    if sha256_file(final['image']) != final.get('sha256') or (image is not None and sha256_file(image) != final['sha256']):
        raise ValueError('最终图摘要不符')
    if not np.array_equal(read_rgb(final['image']), read_rgb(ready['generative_anchor'])):
        raise ValueError('最终图出现辅助像素加性残差')
    return ready


validate_complete = validate_distributed_complete


def generate_distributed_no_pixel(source, target, output, *, definition, prefix_root,
                                  source_public_record, profile='distributed_balanced',
                                  device='cuda', pause_path=None):
    if profile not in DISTRIBUTED_PRESETS:
        raise ValueError('未知区域分布档位')
    config = DistributedLatentConfig(**DISTRIBUTED_PRESETS[profile]); config.validate()
    prefix, original_ready, data, _, _, prefix_binding = validate_prefix(prefix_root, definition)
    if any(sha256_file(given) != sha256_file(data[role]) for role, given in (('source', source), ('target', target))):
        raise ValueError('调用公开图对不符')
    record = _canonical(source_public_record)
    source_key = build_source_public_key(record, data['source_title'])
    root = Path(output).resolve(); root.relative_to(PROJECT.resolve()); root.mkdir(parents=True, exist_ok=True)
    pause = Path(pause_path) if pause_path else root/'PAUSE_REQUESTED'
    if pause.exists():
        raise InterruptedError('本轮已请求暂停')
    torch.manual_seed(config.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(config.seed)
    from src.visual.encoders.clip import make_encoder
    base = make_encoder(device=device); surrogate = base.surrogate; device = surrogate.device
    vae = _VaeOnly(device)
    implementation = _files([Path(__file__), Path(__file__).with_name('__init__.py')]
        + [PROJECT/p for p in ('src/visual/_perlage_no_pixel_rankfocus/pipeline.py',
            'src/visual/_perlage_no_pixel_rankfocus/__init__.py', 'src/visual/_perlage/components/core.py',
            'src/visual/encoders/backbone.py', 'src/visual/encoders/clip.py', 'src/visual/encoders/joint.py',
            'src/visual/robust_objectives.py', 'src/visual/retrieval_objective.py')])
    identity = _canonical(dict(method=METHOD, definition=DEFINITION, profile=profile,
        config=asdict(config), public_definition=definition, source_public_record=record,
        source_public_key_text=source_key, prefix_binding=prefix_binding, inputs=_files([source,target]),
        encoder=surrogate.snapshot(), vae=vae.identity, implementation=implementation,
        training_data_policy=TRAINING_DATA_POLICY, training_queries=data['training_queries'],
        objective='equal_fullkey_joint_and_compensated_cosine_minus_region_normalized_visual_cost',
        mask_policy='foreground_background_separate_budgets_print_and_two_pixel_boundary_protected',
        pixel_optimization=False, auxiliary_levels=0, diffusion_calls=0,
        private_gallery_used=False, victim_memory_used=False, formal_queries_used=False,
        shadow_pool_used=False, true_rank_used=False, adaptive_stopping='public_utility_only'))
    identity_path = root/'method_identity.json'
    if identity_path.exists():
        if _read(identity_path) != identity: raise ValueError('区域分布实验身份改变，须另建目录')
    elif any(root.iterdir()):
        raise ValueError('拒绝覆盖没有本轮身份的非空目录')
    else: write_json(identity_path, identity)
    if (root/'ready.json').exists():
        ready=validate_distributed_complete(_read(root/'ready.json'))
        del base,vae;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
        return Path(ready['candidates']['final']['image']),ready
    tick=time.perf_counter()
    original=read_tensor(source,device)
    anchor_path=prefix/'color_harmonization/generative_anchor.png'
    anchor=read_tensor(anchor_path,device)
    foreground=read_rgb(prefix/'foreground_segmentation/source/foreground.png')[...,0]>0
    protected=print_region_mask(read_rgb(source).shape,definition['print_regions'])
    budget_array=build_budget_map(foreground,protected,config)
    write_json(root/'regional_budget.json',dict(foreground_levels=config.foreground_levels,
        background_levels=config.background_levels,reference='frozen_color_harmonization_anchor',
        protected_pixels=int(protected.sum()),foreground_pixels=int(foreground.sum()),
        background_pixels=int((~foreground).sum())))
    Image.fromarray(np.rint(budget_array/max(config.foreground_levels,1)*255).astype(np.uint8)).save(root/'latent_edit_mask.png')
    budgets=torch.as_tensor(budget_array,device=device,dtype=torch.float32)[None,None]
    fg=_mask_tensor(foreground,original); bg=1-fg
    with torch.no_grad():target_feature=surrogate.encode_image(read_tensor(target,device)).global_embedding.detach()
    joint=PublicFullKeyObjective(surrogate,target_feature,source_key,data['training_queries'])
    write_json(root/'public_title_encoding_audit.json',joint.audit)
    branch=_SpatialLatentBranch(anchor,budgets,vae,config)
    optimizer=torch.optim.Adam([branch.delta],lr=config.latent_learning_rate)

    def compose():return quantize_ste((anchor+branch.residual()).clamp(0,1))

    def visual_cost(image):
        delta=(image-original)/(config.foreground_levels/255)
        l2=sum(_region_mean(delta.square(),m) for m in (fg,bg))
        dy=delta[...,1:,:]-delta[...,:-1,:];dx=delta[...,:,1:]-delta[...,:,:-1]
        tv=image.sum()*0
        for m in (fg,bg):
            my=torch.minimum(m[...,1:,:],m[...,:-1,:]);mx=torch.minimum(m[...,:,1:],m[...,:,:-1])
            tv+=_region_mean(dy.abs(),my)+_region_mean(dx.abs(),mx)
        blur=F.avg_pool2d(F.pad(delta,(2,2,2,2),mode='reflect'),5,stride=1)
        texture=sum(_region_mean((delta-blur).square(),m) for m in (fg,bg))
        return config.deviation_l2_weight*l2+config.residual_tv_weight*tv+config.texture_weight*texture

    def value_and_penalty(image):
        features=surrogate.encode_image(image).global_embedding
        scores,comp=joint.scores(features),joint.compensated_scores(features)
        utility=.25*(scores.mean()+softmin(scores.flatten(),config.temperature)
                     +comp.mean()+softmin(comp.flatten(),config.temperature))
        return utility,visual_cost(image)

    @torch.no_grad()
    def development(image,step):
        q={k:float(v) for k,v in region_quality(image,original,anchor,foreground,config).items()}
        u,p=value_and_penalty(image)
        q.update(step=step,training_utility=float(u),regularization=float(p),selection_utility=float(u-p))
        q.update(joint.diagnostics(surrogate.encode_image(image).global_embedding))
        q['quality_ok']=bool(_quality_ok(q,config));q['selection']=[int(q['quality_ok']),q['selection_utility']]
        return q

    history=[];best_row=None;best=None;step_done=0;previous=0.;accepted=rejected=0
    checkpoint=root/'checkpoint.pt'
    if checkpoint.exists():
        state=torch.load(relocate_io_path(checkpoint),map_location=device,weights_only=False)
        if state['identity']!=identity:raise ValueError('区域分布断点身份改变')
        with torch.no_grad():branch.delta.copy_(state['latent'])
        optimizer.load_state_dict(state['optimizer'])
        history,best_row,best=state['history'],state['best_row'],state['best']
        step_done,previous=state['step'],state['seconds'];accepted,rejected=state['accepted'],state['rejected']
        # 暂停可能在检查点之间；停止证书只由已经完成的完整检查点重算。
        if history and history[-1]['step']>step_done:raise ValueError('断点历史越过实际步数')
        if state.get('early_stop')!=update_early_stopping(history,config):raise ValueError('断点停止状态改变')

    def save(status):
        stop=update_early_stopping(history,config)
        elapsed=previous+time.perf_counter()-tick
        pending=checkpoint.with_suffix('.tmp')
        torch.save(dict(identity=identity,latent=branch.delta.detach(),optimizer=optimizer.state_dict(),
            history=history,best_row=best_row,best=best,step=step_done,seconds=elapsed,
            accepted=accepted,rejected=rejected,early_stop=stop),relocate_io_path(pending))
        pending.replace(checkpoint)
        write_json(root/'progress.json',dict(status=status,profile=profile,step=step_done,
            requested_steps=config.steps,best=best_row,accepted_updates=accepted,rejected_updates=rejected,
            seconds=elapsed,early_stop=stop))

    if best_row is None:
        best=branch.delta.detach().clone();best_row=development(compose(),0);history.append(best_row)
        if not best_row['quality_ok']:
            write_json(root/'initial_quality_rejection.json',best_row)
            raise ValueError('冻结起点未通过预注册区域质量门槛')
        save('running')
    if not update_early_stopping(history,config)['rule_satisfied']:
      for step in range(step_done+1,config.steps+1):
        if pause.exists():save('paused');raise InterruptedError('区域分布实验已保存断点')
        image=compose().detach().requires_grad_(True)
        metrics=region_quality(image,original,anchor,foreground,config)
        constraints=[(metrics['source_ssim'],config.source_ssim_floor),
            (metrics['foreground_ssim'],config.foreground_ssim_floor),
            (-metrics['foreground_mae_levels']/255,-config.foreground_mae_ceiling/255),
            (-metrics['background_mae_levels']/255,-config.background_mae_ceiling/255),
            (-metrics['background_added_mae_levels']/255,-config.background_added_mae_ceiling/255)]
        normals=[];slacks=[]
        for value,floor in constraints:
            normal,=torch.autograd.grad(value,image,retain_graph=True)
            normals.append(normal.detach());slacks.append(float(value.detach())-floor)
        utility,penalty=value_and_penalty(image)
        gradient,=torch.autograd.grad(-utility+penalty,image)
        if not torch.isfinite(gradient).all():raise FloatingPointError('区域图像梯度非有限')
        old=branch.delta.detach().clone();old_optimizer=copy.deepcopy(optimizer.state_dict())
        optimizer.zero_grad(set_to_none=True);decoded=compose();latent_normals=[]
        for normal in normals:
            derivative,=torch.autograd.grad(decoded,branch.delta,grad_outputs=normal,retain_graph=True)
            latent_normals.append(derivative.detach())
        decoded.backward(gradient)
        if branch.delta.grad is None or not torch.isfinite(branch.delta.grad).all():raise FloatingPointError('区域潜变量梯度非有限')
        optimizer.param_groups[0]['lr']=config.latent_learning_rate*(1-.75*step/config.steps)
        optimizer.step()
        with torch.no_grad():
            branch.delta.clamp_(-config.latent_limit,config.latent_limit)
            smooth=F.avg_pool2d(F.pad(branch.delta,(1,1,1,1),mode='reflect'),3,stride=1)
            branch.delta.lerp_(smooth,config.latent_smoothing_mix)
            proposed=(old+tangent_projection(branch.delta-old,latent_normals,slacks)).clamp(-config.latent_limit,config.latent_limit)
            fraction=0.
            for backtrack in range(config.guard_backtracks+1):
                scale=.5**backtrack;branch.delta.copy_(old+scale*(proposed-old))
                q={k:float(v) for k,v in region_quality(compose(),original,anchor,foreground,config).items()}
                if _quality_ok(q,config):fraction=scale;break
            if fraction:accepted+=1
            else:branch.delta.copy_(old);optimizer.load_state_dict(old_optimizer);rejected+=1
        step_done=step
        if step%config.check_every==0:
            row=development(compose(),step);history.append(row)
            save_tensor_image(root/f'checkpoint_{step:04d}.png',compose())
            if tuple(row['selection'])>tuple(best_row['selection']):best,best_row=branch.delta.detach().clone(),row
            save('running')
            print(json.dumps(dict(profile=profile,step=step,best_step=best_row['step'],
                utility=row['training_utility'],selection_utility=row['selection_utility'],
                foreground_ssim=row['foreground_ssim'],foreground_mae=row['foreground_mae_levels'],
                background_mae=row['background_mae_levels'],accepted=accepted,rejected=rejected,
                stop=update_early_stopping(history,config)),ensure_ascii=False),flush=True)
            if update_early_stopping(history,config)['rule_satisfied']:break
    early=update_early_stopping(history,config)
    if not early['rule_satisfied']:raise AssertionError('没有完成预注册自适应停止日程')
    with torch.no_grad():branch.delta.copy_(best);final_tensor=compose();final_check=development(final_tensor,best_row['step'])
    if not final_check['quality_ok']:raise AssertionError('最终区域质量守门失败')
    final=root/'adversarial.png';generative_anchor=root/'generative_anchor.png'
    save_tensor_image(final,final_tensor);save_tensor_image(generative_anchor,final_tensor)
    arr=read_rgb(final);old_arr=read_rgb(anchor_path);change=np.abs(arr.astype(np.int16)-old_arr.astype(np.int16))
    if (np.any(change[protected]) or change[foreground].max()>config.foreground_levels
            or change[~foreground].max()>config.background_levels):
        raise AssertionError('区域预算或印刷保护失败')
    metadata=dict(status='completed',config=asdict(config),steps_completed=step_done,
        history=history,best=best_row,final_check=final_check,early_stop=early,stop_reason=early['reason'],
        accepted_updates=accepted,rejected_updates=rejected,foreground_max_levels=int(change[foreground].max()),
        background_max_levels=int(change[~foreground].max()),latent_max_levels=int(change.max()),
        latent_mae_levels=float(change.mean()),auxiliary_max_levels=0,auxiliary_mae_levels=0.,
        pixel_optimization=False,no_pixel_optimization=True,diffusion_calls=0,
        foreground_limited=config.background_levels==0,background_optimization=config.background_levels>0,
        printed_region_fixed_to_harmonized_anchor=True,private_gallery_used=False,true_rank_used=False,
        elapsed_seconds=previous+time.perf_counter()-tick,output_sha256=sha256_file(final))
    write_json(root/'metadata.json',metadata)
    stages=copy.deepcopy(prefix_binding['stages'])
    for record_stage in stages.values():record_stage['execution_mode']='reused_frozen_prefix'
    stages[STAGES[-1]]=stage_record([identity_path,root/'metadata.json',final,generative_anchor,
        root/'latent_edit_mask.png',root/'regional_budget.json',root/'public_title_encoding_audit.json'],
        steps_completed=step_done,steps_requested=config.steps,schedule='adaptive_public_plateau_v1',
        execution_mode='fresh_adaptive_refinement')
    ready=dict(method=METHOD,definition=DEFINITION,profile=profile,status='completed',color_scheme='statistics',
        training_data_policy=TRAINING_DATA_POLICY,first_three_stages_reused=True,fresh_first_three_stages_executed=False,
        initialization_reused=True,previous_final_refinement_reused=False,method_identity=str(identity_path),
        method_identity_sha256=sha256_file(identity_path),metadata=str(root/'metadata.json'),
        source_sha256=sha256_file(source),target_sha256=sha256_file(target),
        source_title_sha256=original_ready['source_title_sha256'],stages=stages,latent_config=asdict(config),
        source_public_key_text=source_key,candidates={'final':dict(image=str(final),sha256=sha256_file(final))},
        generative_anchor=str(generative_anchor),prefix_binding=prefix_binding,early_stop=early,
        pixel_optimization=False,auxiliary_max_levels=0,optimization_mode='latent_only',
        victim_memory_used=False,formal_queries_used=False,shadow_pool_used=False,true_rank_used=False,
        purchase_history_used=False,diffusion_calls=0,observed_seconds=metadata['elapsed_seconds'])
    validate_distributed_complete(ready,final);write_json(root/'ready.json',ready);save('completed')
    del base,vae,surrogate,branch,optimizer;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return final,ready
