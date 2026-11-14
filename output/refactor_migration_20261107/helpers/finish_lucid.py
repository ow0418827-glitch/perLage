from pathlib import Path
import ast
import json

root = Path(__file__).resolve().parents[1]
original = (root / 'tmp/pre_refactor/src/vendor/dual_rag/optimization/common/config.py').read_text(encoding='utf-8')
tree = ast.parse(original)
classes = []
for node in tree.body:
    if isinstance(node, ast.ClassDef) and node.name in ['AttackConfig', 'SurrogateConfig', 'RuntimeConfig']:
        classes.append(ast.get_source_segment(original, node))
header = '''"""原始像素优化配置；不包含频域、选图、受害记忆或购物配置。"""
from dataclasses import asdict, dataclass, field, replace
import json
import math
from typing import Any, Mapping
from src.paths import ProjectPath as Path, VISUAL_CONFIGS

'''
for item in classes:
    header += '@dataclass(frozen=True)\n' + item + '\n\n'
(root / 'src/visual/_lucid/config.py').write_text(header + '''
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
    from .surrogates import resolve_hf_cached_weight
    from .model_paths import local_model_candidates
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
''', encoding='utf-8')
path = root / 'src/visual/config/optimization.json'
raw = json.loads(path.read_text(encoding='utf-8'))
for name in ['frequency', 'selection', 'shared_engineering']:
    raw.pop(name, None)
raw['runtime']['cache_directory'] = 'output/cache'
path.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

# Shared encoder imports no longer point through the removed vendor package.
path = root / 'src/visual/encoders/clip.py'
body = path.read_text(encoding='utf-8').replace('from .io import digest', 'from src.visual.io import sha256_file as digest')
body = body.replace('from .model_paths import resolve_memory_model', 'from .._lucid.model_paths import resolve_memory_model')
body = body.replace('from .optimization.common.config import SurrogateConfig', 'from .._lucid.config import SurrogateConfig')
body = body.replace('from .optimization.attack_core.surrogates import OpenCLIPSurrogate', 'from .._lucid.surrogates import OpenCLIPSurrogate')
path.write_text(body, encoding='utf-8')
path = root / 'src/visual/_lucid/model_paths.py'
path.write_text(path.read_text(encoding='utf-8').replace('from .optimization.attack_core.surrogates', 'from .surrogates'), encoding='utf-8')
# Baseline text targets are optional; the missing upstream audit module is replaced by our strict title adapter.
path = root / 'src/visual/_lucid/surrogates.py'
text = path.read_text(encoding='utf-8')
text = text.replace('from src.visual._lucid.config import SurrogateConfig', 'from .config import SurrogateConfig')
path.write_text(text, encoding='utf-8')
