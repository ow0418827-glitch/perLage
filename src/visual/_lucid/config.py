"""原始像素优化配置；不包含频域、选图、受害记忆或购物配置。"""
from dataclasses import asdict, dataclass, field, replace
import json
import math
from typing import Any, Mapping
from src.paths import ProjectPath as Path, VISUAL_CONFIGS

@dataclass(frozen=True)
class AttackConfig:
    epsilon: float = 16 / 255
    spatial_step_size: float = 0.5 / 255
    reference_steps: int = 1000
    initialization: str = "zero"
    clusters: int = 10
    cluster_iterations: int = 100
    sinkhorn_iterations: int = 100
    sinkhorn_regularization: float = 0.1
    sinkhorn_tolerance: float = 1e-2
    detach_transport_plan: bool = True
    local_weight: float = 0.2
    text_weight: float = 1.0
    text_model_index: int = 1
    log_every: int = 50
    model_weight_update: str = "softmax_negative_preupdate_global_similarity"

    def validate(self) -> None:
        if not math.isfinite(self.epsilon) or not 0 < self.epsilon <= 1:
            raise ValueError("attack.epsilon 必须位于 (0, 1]。")
        if not math.isfinite(self.spatial_step_size) or self.spatial_step_size <= 0:
            raise ValueError("attack.spatial_step_size 必须为正数。")
        if self.reference_steps < 0:
            raise ValueError("attack.reference_steps 不能为负数。")
        if self.initialization != "zero":
            raise ValueError("兼容实验只允许零初始化。")
        if self.clusters <= 0 or self.cluster_iterations <= 0:
            raise ValueError("聚类数和聚类迭代数必须为正整数。")
        if (
            self.sinkhorn_iterations <= 0
            or not math.isfinite(self.sinkhorn_regularization)
            or self.sinkhorn_regularization <= 0
        ):
            raise ValueError("最优传输迭代数和正则系数必须为正数。")
        if not math.isfinite(self.sinkhorn_tolerance) or self.sinkhorn_tolerance < 0:
            raise ValueError("最优传输停止容差不能为负数。")
        if (
            not math.isfinite(self.local_weight)
            or not math.isfinite(self.text_weight)
            or self.local_weight < 0
            or self.text_weight < 0
        ):
            raise ValueError("目标函数权重不能为负数。")
        if self.log_every <= 0:
            raise ValueError("attack.log_every 必须为正整数。")

from ..encoders.backbone import BackboneConfig as SurrogateConfig

@dataclass(frozen=True)
class RuntimeConfig:
    allow_downloads: bool = False
    allow_device_fallback: bool = False
    require_true_local_tokens: bool = True
    hash_model_weights: bool = True
    cache_directory: str = "outputs/cache"
    device_memory_sample_interval_seconds: float = 0.05

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "RuntimeConfig":
        return cls(
            allow_downloads=bool(raw.get("allow_downloads", False)),
            allow_device_fallback=bool(raw.get("allow_device_fallback", False)),
            require_true_local_tokens=bool(raw.get("require_true_local_tokens", True)),
            hash_model_weights=bool(raw.get("hash_model_weights", True)),
            cache_directory=str(raw.get("cache_directory", "outputs/cache")),
            device_memory_sample_interval_seconds=float(raw.get("device_memory_sample_interval_seconds", 0.05)),
        )

    def validate(self) -> None:
        if (
            not math.isfinite(self.device_memory_sample_interval_seconds)
            or self.device_memory_sample_interval_seconds <= 0
        ):
            raise ValueError("设备显存采样间隔必须为有限正数。")
        if not self.cache_directory.strip():
            raise ValueError("运行缓存目录不能为空。")


@dataclass(frozen=True)
class AcceptanceConfig:
    decoded_max_pixel_difference: float = 16


@dataclass(frozen=True)
class ProjectConfig:
    attack: AttackConfig
    surrogates: tuple[SurrogateConfig, ...]
    seed: int = 42
    device: str = 'cuda'
    attack_precision: str = 'fp32'
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    acceptance: AcceptanceConfig = field(default_factory=AcceptanceConfig)

    def validate(self):
        self.attack.validate()
        self.runtime.validate()
        if len(self.surrogates) != 3 or not 0 <= self.attack.text_model_index < 3:
            raise ValueError('像素优化需要三个代理及有效的文字代理序号')
        if self.attack_precision != 'fp32' or self.runtime.allow_device_fallback:
            raise ValueError('正式像素优化要求单精度且禁止设备回退')
        if not math.isfinite(self.acceptance.decoded_max_pixel_difference) or self.acceptance.decoded_max_pixel_difference <= 0:
            raise ValueError('落盘像素预算必须为有限正数')

    def to_dict(self):
        return asdict(self)


def load_config(path=None, *, validate=True):
    path = Path(path or VISUAL_CONFIGS / 'optimization.json').resolve()
    raw = json.loads(path.read_text(encoding='utf-8'))
    if 'frequency' in raw:
        raise ValueError('频域配置已经移除，请使用纯像素配置')
    value = ProjectConfig(
        attack=AttackConfig(**raw.get('attack', {})),
        surrogates=tuple(SurrogateConfig.from_mapping(r) for r in raw['surrogates']),
        seed=raw.get('seed', 42), device=raw.get('device', 'cuda'),
        attack_precision=raw.get('attack_precision', 'fp32'),
        runtime=RuntimeConfig.from_mapping(raw.get('runtime', {})),
        acceptance=AcceptanceConfig(**raw.get('acceptance', {})))
    if validate:
        value.validate()
    return value


def resolve_optimization(path=None, surrogate_indices=None):
    from ..encoders.backbone import resolve_hf_cached_weight
    from ..encoders.model_paths import local_model_candidates
    path = Path(path or VISUAL_CONFIGS / 'optimization.json').resolve()
    config = load_config(path)
    raw = json.loads(path.read_text(encoding='utf-8'))
    roots = [Path(p).expanduser() for p in raw.get('model_search_roots', [])]
    roots = [p if p.is_absolute() else path.parent / p for p in roots]
    models = []
    for index, spec in enumerate(config.surrogates):
        if surrogate_indices is not None and index not in surrogate_indices:
            continue
        weight = Path(spec.weight_path).expanduser() if spec.weight_path else resolve_hf_cached_weight(spec.hf_repo)
        if weight is None and spec.hf_repo:
            weight = next((d / 'open_clip_model.safetensors' for d in local_model_candidates(spec.hf_repo, roots)
                           if (d / 'open_clip_model.safetensors').is_file()), None)
        if weight is not None and not weight.is_absolute():
            weight = path.parent / weight
        if weight is None or not weight.is_file():
            raise FileNotFoundError(f'缺少代理权重：{spec.name}/{spec.pretrained}')
        models.append(replace(spec, weight_path=str(weight.resolve())))
    return replace(config, surrogates=tuple(models), attack=replace(config.attack, text_weight=0.0))
