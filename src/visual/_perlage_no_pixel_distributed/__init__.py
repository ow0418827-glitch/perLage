"""主体与背景分布式潜变量实验；不改变既有方法接口。"""
from .pipeline import (
    METHOD, DEFINITION, DISTRIBUTED_PRESETS, PRESETS, DistributedLatentConfig,
    build_budget_map, region_quality, update_early_stopping,
    generate_distributed_no_pixel, validate_distributed_complete,
)

validate_complete = validate_distributed_complete

__all__ = ['METHOD', 'DEFINITION', 'DISTRIBUTED_PRESETS', 'PRESETS',
           'DistributedLatentConfig', 'build_budget_map', 'region_quality',
           'update_early_stopping', 'generate_distributed_no_pixel',
           'validate_distributed_complete', 'validate_complete']
