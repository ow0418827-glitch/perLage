"""本轮排名优先试验专用接口，不修改原无像素方法。"""
from .pipeline import (
    METHOD, PRESETS, RANKFOCUS_PRESETS, RankFocusLatentConfig,
    build_source_public_key, generate_rankfocus_no_pixel,
    validate_complete, validate_prefix,
)
validate_rankfocus_complete = validate_complete
__all__ = ['METHOD', 'PRESETS', 'RANKFOCUS_PRESETS', 'RankFocusLatentConfig',
           'build_source_public_key', 'generate_rankfocus_no_pixel',
           'validate_complete', 'validate_rankfocus_complete', 'validate_prefix']
