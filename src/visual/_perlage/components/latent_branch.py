"""供保留方法共用的基础工具；不提供其他图片优化方法。"""
from __future__ import annotations

import torch
import torch.nn.functional as F

class SourceLatentBranch:
    """零潜变量精确返回零编辑，消除自编码重建本身引入的源图漂移。"""
    def __init__(self, source, mask, purifier, config):
        self.source, self.mask, self.purifier, self.config = source, mask, purifier, config
        with torch.no_grad():
            from src.visual.diffusion_purification import working_shape
            work = F.interpolate(source, size=working_shape(source.shape, config.latent_working_size),
                mode='bicubic', align_corners=False, antialias=True).clamp(0, 1)
            self.base = (purifier.vae.encode(work * 2 - 1).latent_dist.mode().float()
                * purifier.vae.config.scaling_factor).detach()
            self.reference = purifier._decode(self.base, source.shape).detach()
        self.delta = torch.zeros_like(self.base, requires_grad=True)

    def residual(self):
        eps = self.config.latent_levels / 255
        decoded = self.purifier._decode(self.base + self.delta, self.source.shape)
        return self.mask * eps * torch.tanh((decoded - self.reference) / eps)
