"""供保留方法共用的基础工具；不提供其他图片优化方法。"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from src.visual.image_utils import quantize_ste
from src.visual.defense_transforms import gaussian_blur_bpda, differentiable_jpeg_ste

def softmin(values, temperature):
    values = values.reshape(-1)
    return -temperature * (torch.logsumexp(-values / temperature, 0) - math.log(values.numel()))


def transformed_batch(image, *, step=None):
    """真实量化、四档模糊、四档压缩及缩放；原预处理代理和真实前向不变。"""
    x = quantize_ste(image)
    blur_radii, jpeg_qualities = (.5, 1., 2., 4.), (90, 75, 50, 25)
    if step is not None:
        i = (step - 1) % 4
        return torch.cat([x, gaussian_blur_bpda(x, blur_radii[i]),
            differentiable_jpeg_ste(x, jpeg_qualities[i], subsampling=2, exact_forward=True)])
    from src.visual.diffusion_purification import working_shape
    resized = F.interpolate(F.interpolate(x, size=working_shape(x.shape, 512), mode='bicubic', align_corners=False,
        antialias=True), size=x.shape[-2:], mode='bicubic', align_corners=False, antialias=True).clamp(0, 1)
    return torch.cat([x] + [gaussian_blur_bpda(x, r) for r in blur_radii]
        + [differentiable_jpeg_ste(x, q, subsampling=2, exact_forward=True) for q in jpeg_qualities] + [quantize_ste(resized)])


def compensated_targets(queries, history):
    """等权单位图文融合的解析逆像；非正半球的查询不采用该教师。

    若单位向量 q 与 h 的内积为正，v=2(q·h)q-h 为单位向量，且
    normalize(v+h)=q。这里只能说明表示空间目标，不能保证图像可达性。
    """
    if queries.ndim != 2 or history.ndim != 2 or history.shape[0] != 1 or queries.shape[1] != history.shape[1]:
        raise ValueError('自身历史和查询特征尺寸无效')
    if not torch.isfinite(queries).all() or not torch.isfinite(history).all() or (queries.norm(dim=-1) == 0).any() or history.norm() == 0:
        raise ValueError('特征必须是有限非零向量')
    q = F.normalize(queries.detach().float(), dim=-1)
    h = F.normalize(history.detach().float(), dim=-1)
    dot = q @ h.T
    return F.normalize(2 * dot * q - h, dim=-1), dot[:, 0] > 1e-4


def weakest_timestep(step, rows, timesteps):
    """四步中三步关注开发效用最弱档，一步覆盖其余档；不读排名。"""
    if step % 4 == 0:
        return timesteps[(step // 4 - 1) % len(timesteps)]
    return min(timesteps, key=lambda t: min(r['utility'] for r in rows if r['timestep'] == t))


def feasible_fraction(check, maximum_backtracks):
    """返回通过实际代理保留门槛的最大回退步幅；全失败返回零。"""
    for index in range(maximum_backtracks + 1):
        fraction = 0.5 ** index
        if check(fraction):
            return fraction
    return 0.
