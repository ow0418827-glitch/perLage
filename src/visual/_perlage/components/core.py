"""完整感知方法的内部组件；不提供其他方法入口。"""
from __future__ import annotations
from dataclasses import asdict,dataclass
import itertools
import math
import torch
import torch.nn.functional as F

@dataclass(frozen=True)
class NaturalInitializationConfig:
    latent_clean_steps: int = 40
    latent_diffusion_steps: int = 40
    pixel_clean_steps: int = 20
    pixel_diffusion_steps: int = 60
    latent_levels: int = 32
    latent_working_size: int = 256
    latent_limit: float = 0.5
    latent_learning_rate: float = 0.012
    latent_smoothing_mix: float = 0.1
    pixel_levels: int = 4
    pixel_start_levels: float = 0.5
    pixel_end_levels: float = 0.08
    momentum: float = 0.85
    temperature: float = 0.04
    teacher_weight: float = 0.5
    semantic_image_weight: float = 1.0
    semantic_joint_weight: float = 0.5
    semantic_teacher_weight: float = 0.05
    image_weight: float = 0.05
    diffusion_weight: float = 3.0
    shadow_weight: float = 0.0
    guard_tolerance: float = 0.01
    latent_guard_tolerance: float = 0.04
    development_temperature: float = 0.01
    guard_backtracks: int = 4
    source_ssim_floor: float = 0.93
    ordinary_ssim_floor: float = 0.95
    source_mae_ceiling: float = 6.0
    ssim_safety_buffer: float = 1e-05
    tv_weight: float = 0.003
    l2_weight: float = 0.001
    check_every: int = 20
    full_eot_samples: int = 1
    diffusion_timesteps: tuple[int, ...] = (99, 199, 399)
    validation_seeds: tuple[int, ...] = (9301, 9302)
    training_seed_base: int = 710000000

    def validate(self):
        if isinstance(self.check_every, bool) or not isinstance(self.check_every, int) or self.check_every < 1:
            raise ValueError('检查间隔必须为正整数')
        for name in ('latent_clean_steps', 'latent_diffusion_steps', 'pixel_clean_steps', 'pixel_diffusion_steps'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value % self.check_every:
                raise ValueError('阶段步数必须非负且具有完整检查点')
        if self.check_every < 1 or self.latent_clean_steps == 0:
            raise ValueError('生成阶段必须先完成正常条件初始化')
        if not 0 <= self.pixel_levels <= 4 or not 1 <= self.latent_levels <= 32:
            raise ValueError('像素收尾至多四级；生成细化预算至多三十二级')
        if not 0 < self.latent_learning_rate or not 0 < self.latent_limit or (not 0 <= self.latent_smoothing_mix <= 1):
            raise ValueError('生成参数无效')
        if not 0 <= self.momentum < 1 or not 0 < self.pixel_end_levels <= self.pixel_start_levels:
            raise ValueError('像素参数无效')
        if not 0.9 <= self.source_ssim_floor <= 1 or not 0 < self.source_mae_ceiling <= 10:
            raise ValueError('图像质量限制无效')
        if not self.source_ssim_floor <= self.ordinary_ssim_floor <= 0.99:
            raise ValueError('普通阶段质量预留门槛无效')
        if isinstance(self.full_eot_samples, bool) or self.full_eot_samples not in (1, 2) or (not self.validation_seeds) or (not self.diffusion_timesteps):
            raise ValueError('真实净化路径日程无效')
        if not 0 <= self.guard_tolerance <= self.latent_guard_tolerance <= 0.05 or not 0 < self.development_temperature <= 0.1:
            raise ValueError('阶段保留容差或公开边际无效')
        if any((not math.isfinite(v) for v in asdict(self).values() if isinstance(v, float))):
            raise ValueError('参数必须有限')

def source_ssim(image, source):
    """七像素均匀窗口、样本协方差，与默认结构相似度统计约定一致。"""
    image, source = (image.double(), source.double())
    mu_x, mu_y = (F.avg_pool2d(image, 7, stride=1), F.avg_pool2d(source, 7, stride=1))
    covariance = 49 / 48
    vx = covariance * (F.avg_pool2d(image.square(), 7, stride=1) - mu_x.square())
    vy = covariance * (F.avg_pool2d(source.square(), 7, stride=1) - mu_y.square())
    vxy = covariance * (F.avg_pool2d(image * source, 7, stride=1) - mu_x * mu_y)
    return ((2 * mu_x * mu_y + 0.01 ** 2) * (2 * vxy + 0.03 ** 2) / ((mu_x.square() + mu_y.square() + 0.01 ** 2) * (vx + vy + 0.03 ** 2))).mean()

def pixel_projection(anchor, pixel, mask, levels):
    bound = torch.floor(mask * levels + 1e-06) / 255
    return torch.maximum(torch.minimum(pixel, torch.minimum(bound, 1 - anchor)), torch.maximum(-bound, -anchor))

def tangent_projection(direction, normals, slacks):
    """至多五个普通条件及质量线性约束的活跃集投影；真实非线性约束仍须回退检验。



    求 min ||d-p||²/2，约束 a_i·d >= -slack_i/2。

    枚举少量活跃约束，用双精度小矩阵求解；病态时回退零更新。

    """
    if not normals:
        return direction
    shape = direction.shape
    a = torch.stack([n.detach().flatten() for n in normals])
    norms = a.norm(dim=-1)
    valid = norms > 1e-12
    if not valid.any():
        return direction
    a = a[valid] / norms[valid, None]
    lower = -0.5 * direction.new_tensor(slacks)[valid].clamp_min(0) / norms[valid]
    gram = (a.double() @ a.double().T).cpu()
    current = (a.double() @ direction.flatten().double()).cpu()
    rhs = lower.double().cpu()
    best_cost = float('inf')
    best = None
    for count in range(len(a) + 1):
        for active in itertools.combinations(range(len(a)), count):
            if not active:
                coefficients = torch.zeros(len(a), dtype=torch.float64)
            else:
                idx = list(active)
                g = gram[idx][:, idx]
                value = torch.linalg.pinv(g, rtol=1e-10) @ (rhs - current)[idx]
                if (value < -1e-08).any():
                    continue
                coefficients = torch.zeros(len(a), dtype=torch.float64)
                coefficients[idx] = value.clamp_min(0)
            if (current + gram @ coefficients < rhs - 1e-08).any():
                continue
            cost = float(coefficients @ gram @ coefficients)
            if cost < best_cost:
                best_cost, best = (cost, coefficients)
    if best is None:
        return torch.zeros_like(direction)
    result = direction.flatten() + best.to(a) @ a
    return result.reshape(shape)

