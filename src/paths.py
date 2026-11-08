"""目录职责与本项目历史路径迁移；不重写任何相邻项目路径。"""
import os
from pathlib import Path as _Path

PROJECT = _Path(__file__).resolve().parents[1]
SOURCE = PROJECT / 'src'
OUTPUT = PROJECT / 'output'
DATA = PROJECT / 'data'
CONFIGS = SOURCE / 'agent/config'
PROMPTS = SOURCE / 'agent/prompts/templates'
VISUAL_CONFIGS = SOURCE / 'visual/config'
EXPERIMENT_CONFIGS = SOURCE / 'experiments/config'
PUBLIC = OUTPUT / 'datasets/public'
PRIVATE = OUTPUT / 'datasets/private'
SHARED_MEMORY = OUTPUT / 'shared_memory/pool_189_v1/shared_memory.json'
MODEL_ROOT = _Path(os.environ.get('PERLAGE_MODEL_ROOT', PROJECT / 'models')).resolve()

# 清单和完成证据保持原始字节；只在实际文件访问时翻译本项目旧位置。
_RELOCATIONS = (
    ('src/output', 'output'),
    ('src/data/public', 'output/datasets/public'),
    ('src/data/private', 'output/datasets/private'),
    ('src/data/indication_prompt_template_neutral.json', 'src/agent/prompts/templates/neutral.json'),
    ('src/configs', 'src/agent/config'),
    ('src/configuration', 'src/agent/config'),
    ('src/memory', 'src/agent/memory'),
    ('src/vision/dual_rag_bridge.py', 'src/visual/_lucid/pipeline.py'),
    ('src/vision/joint_encoder.py', 'src/visual/encoders/joint.py'),
    ('src/vision/joint_encoder_worker.py', 'src/visual/encoders/worker.py'),
    ('src/vision/_natural', 'src/visual/_perlage/components'),
    ('src/vision/natural_visual_edit.py', 'src/visual/_perlage/components/instruction.py'),
    ('src/vision/latent_branch.py', 'src/visual/_perlage/components/latent_branch.py'),
    ('src/vision', 'src/visual'),
    ('src/environment', 'src/webshop'),
    ('src/data_sources', 'src/visual/selection'),
)


def relocate_project_path(value):
    path = _Path(value)
    absolute = path if path.is_absolute() else _Path.cwd() / path
    try:
        relative = absolute.relative_to(PROJECT)
    except ValueError:
        return os.fspath(value)
    for old, new in _RELOCATIONS:
        try:
            tail = relative.relative_to(_Path(old))
        except ValueError:
            continue
        if old == 'src/configs' and tail.parts:
            name = tail.parts[0]
            if name.startswith(('optimization', 'evaluator', 'shape_')):
                new = 'src/visual/config'
            elif name.startswith(('pipeline', 'selection_')):
                new = 'src/experiments/config'
        return str(PROJECT / new / tail)
    return os.fspath(value)


def relocate_io_path(value):
    return relocate_project_path(value) if isinstance(value, (str, os.PathLike)) else value


class ProjectPath(type(_Path())):
    """兼容本项目历史文件位置，保留清单文本与摘要。"""
    def __new__(cls, *args):
        return super().__new__(cls, relocate_project_path(_Path(*args)))


def model_path(name):
    return str(MODEL_ROOT / name)


def public_root_for(value, *, extra_roots=()):
    """固定复现数据、公开缓存或某次实验自己的数据目录。"""
    path = ProjectPath(value).resolve()
    for root in (*extra_roots, DATA, PUBLIC):
        root = ProjectPath(root).resolve()
        if path.is_relative_to(root):
            return root
    if path.is_relative_to(OUTPUT):
        parts = path.relative_to(OUTPUT).parts
        if 'data' in parts:
            return ProjectPath(OUTPUT / _Path(*parts[:parts.index('data') + 1]))
    raise ValueError('公开输入必须位于固定数据、公开缓存或对应实验的数据目录')
