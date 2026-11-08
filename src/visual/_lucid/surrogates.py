"""三代理的加载与调度；编码器实现位于独立共享底座。"""
from typing import Iterable
from ..encoders.backbone import (
    BackboneConfig as SurrogateConfig, OpenCLIPSurrogate, ImageFeatures,
    resolve_device, resolve_hf_cached_weight)


def load_surrogates(
    configs: Iterable[SurrogateConfig],
    *,
    device: str = "cuda",
    precision: str = "fp32",
    allow_downloads: bool = False,
    allow_device_fallback: bool = False,
    require_true_local_tokens: bool = True,
    hash_weights: bool = True,
) -> list[OpenCLIPSurrogate]:
    return [
        OpenCLIPSurrogate(
            config,
            device=device,
            precision=precision,
            allow_downloads=allow_downloads,
            allow_device_fallback=allow_device_fallback,
            require_true_local_tokens=require_true_local_tokens,
            hash_weight=hash_weights,
        )
        for config in configs
    ]
