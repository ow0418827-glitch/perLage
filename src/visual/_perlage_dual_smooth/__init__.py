"""主体与背景双平滑协同潜空间生成感知优化接口。

提供主体曲面余弦内缩平滑微调与极微弱背景余弦过渡平滑漫射的独立生成与验证管线。
原无像素方法及既有实验接口保持只读。
"""
from .pipeline import (
    DUAL_SMOOTH_PRESETS,
    DEFINITION,
    METHOD,
    MASK_POLICY,
    PRESETS,
    DualSmoothLatentConfig,
    build_source_public_key,
    generate_dual_smooth,
    validate_dual_smooth_complete,
    validate_complete,
    validate_prefix,
)

__all__ = [
    'DUAL_SMOOTH_PRESETS',
    'DEFINITION',
    'METHOD',
    'MASK_POLICY',
    'PRESETS',
    'DualSmoothLatentConfig',
    'build_source_public_key',
    'generate_dual_smooth',
    'validate_dual_smooth_complete',
    'validate_complete',
    'validate_prefix',
]
