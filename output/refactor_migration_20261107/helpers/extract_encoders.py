from pathlib import Path
import ast
root=Path(__file__).resolve().parents[1]
config=root/'src/visual/_lucid/config.py'
body=config.read_text(encoding='utf-8')
tree=ast.parse(body)
spec=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SurrogateConfig')
spec_text=ast.get_source_segment(body,spec)
start=min(d.lineno for d in spec.decorator_list)-1
lines=body.splitlines(keepends=True)
body=''.join(lines[:start])+ 'from ..encoders.backbone import BackboneConfig as SurrogateConfig\n'+''.join(lines[spec.end_lineno:])
body=body.replace('from .surrogates import resolve_hf_cached_weight','from ..encoders.backbone import resolve_hf_cached_weight')
body=body.replace('from .model_paths import local_model_candidates','from ..encoders.model_paths import local_model_candidates')
config.write_text(body,encoding='utf-8')
source=root/'src/visual/_lucid/surrogates.py'
old=source.read_text(encoding='utf-8')
prefix,loader=old.split('\ndef load_surrogates(',1)
prefix=prefix.replace('from .config import SurrogateConfig', '@dataclass(frozen=True)\n'+spec_text)
prefix=prefix.replace('SurrogateConfig','BackboneConfig').replace('from typing import Any, Iterable','from typing import Any, Iterable, Mapping')
(root/'src/visual/encoders/backbone.py').write_text(prefix,encoding='utf-8')
source.write_text('''"""三代理的加载与调度；编码器实现位于独立共享底座。"""
from typing import Iterable
from ..encoders.backbone import (
    BackboneConfig as SurrogateConfig, OpenCLIPSurrogate, ImageFeatures,
    resolve_device, resolve_hf_cached_weight)


def load_surrogates('''+loader,encoding='utf-8')
old_path=root/'src/visual/_lucid/model_paths.py'
model_paths=root/'src/visual/encoders/model_paths.py'
old_path.rename(model_paths)
model_paths.write_text(model_paths.read_text(encoding='utf-8').replace('from .surrogates import','from .backbone import'),encoding='utf-8')
p=root/'src/visual/encoders/clip.py'
text=p.read_text(encoding='utf-8').replace('from .._lucid.model_paths import','from .model_paths import')
text=text.replace('from .._lucid.config import SurrogateConfig','from .backbone import BackboneConfig as SurrogateConfig')
text=text.replace('from .._lucid.surrogates import OpenCLIPSurrogate','from .backbone import OpenCLIPSurrogate')
text+='''

def make_encoder(optimization_config=None, surrogate=2, device=None):
    """按公开视觉配置选择编码器；不导入任何优化流水线。"""
    import json
    from dataclasses import asdict, replace
    from src.paths import VISUAL_CONFIGS, PROJECT
    from .backbone import BackboneConfig, resolve_hf_cached_weight
    from .model_paths import local_model_candidates
    if surrogate not in (1,2,3):
        raise ValueError('编码器序号必须是一、二或三')
    path=Path(optimization_config or VISUAL_CONFIGS/'optimization.json').resolve()
    raw=json.loads(path.read_text(encoding='utf-8'))
    if len(raw['surrogates'])!=3:
        raise ValueError('公开代理配置需要三个模型')
    spec=BackboneConfig.from_mapping(raw['surrogates'][surrogate-1])
    weight=Path(spec.weight_path) if spec.weight_path else resolve_hf_cached_weight(spec.hf_repo)
    roots=[Path(p) for p in raw.get('model_search_roots',[])]
    roots=[p if p.is_absolute() else path.parent/p for p in roots]
    if weight is None:
        weight=next((p/'open_clip_model.safetensors' for p in local_model_candidates(spec.hf_repo,roots)
                     if (p/'open_clip_model.safetensors').is_file()),None)
    if weight is not None and not weight.is_absolute():
        weight=path.parent/weight
    if weight is None or not weight.is_file():
        raise FileNotFoundError('缺少公开视觉编码器的本地权重')
    spec=replace(spec,weight_path=str(weight.resolve()))
    return MatchedClipEncoder(dict(surrogate=asdict(spec),surrogate_number=surrogate,
        device=device or raw.get('device','cuda')),PROJECT)
'''
p.write_text(text,encoding='utf-8')
for path in [*(root/'src').rglob('*.py'),*(root/'tools').glob('*.py')]:
    text=path.read_text(encoding='utf-8')
    text=text.replace('from src.visual._lucid.pipeline import make_encoder','from src.visual.encoders.clip import make_encoder')
    path.write_text(text,encoding='utf-8')
# Delete the now-unused memory profiler, plus the duplicated factory in the pixel pipeline.
(root/'src/visual/_lucid/profiling.py').unlink()
p=root/'src/visual/_lucid/pipeline.py';text=p.read_text(encoding='utf-8')
start=text.index('\ndef make_encoder(');end=text.index('\ndef run_attack(',start)
text=text[:start]+text[end:]
p.write_text(text,encoding='utf-8')
