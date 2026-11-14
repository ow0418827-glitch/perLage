"""仅供本轮放宽约束试验使用的独立无像素方法接口。"""
from .pipeline import (
    METHOD,
    RELAXATION_PRESETS,
    RelaxedLatentConfig,
    generate_relaxed_no_pixel,
    validate_complete,
    validate_prefix,
)

validate_relaxed_complete = validate_complete

__all__ = [
    'METHOD', 'RELAXATION_PRESETS', 'RelaxedLatentConfig',
    'generate_relaxed_no_pixel', 'validate_complete', 'validate_relaxed_complete', 'validate_prefix',
]
