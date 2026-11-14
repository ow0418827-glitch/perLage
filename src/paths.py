"""目录职责与本项目历史路径迁移；不重写任何相邻项目路径。"""
import os
from pathlib import Path as _Path, PurePosixPath, PureWindowsPath

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
EXPERIMENT_BASE_GROUPS9 = OUTPUT / 'experiment_base_groups9_20261109'
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


def _original_project_root():
    """迁移只接受显式配置的绝对旧根；不能以磁盘根或文件名猜测项目。"""
    configured = os.environ.get('PERLAGE_ORIGINAL_PROJECT_ROOT')
    if not configured:
        return None
    windows = PureWindowsPath(configured)
    root = windows if windows.is_absolute() else PurePosixPath(configured)
    if not root.is_absolute() or len(root.parts) < 2 or '..' in root.parts:
        raise ValueError('PERLAGE_ORIGINAL_PROJECT_ROOT 必须是无上级跳转的绝对旧项目目录')
    return root


def _map_original_project_path(value):
    """先按旧根自身的平台语义比较，再转换为当前平台的路径。"""
    root = _original_project_root()
    if root is None:
        return None
    original = type(root)(value)
    if not original.is_absolute():
        return None
    try:
        relative = original.relative_to(root)
    except ValueError:
        return None
    tail = []
    for part in relative.parts:
        if part == '..':
            if not tail:
                raise ValueError('旧项目路径的上级跳转超出显式配置的项目根目录')
            tail.pop()
        else:
            # 旧平台的一个文件名不能在当前平台被解释为盘符或多个路径段。
            native = _Path(part)
            if native.anchor or len(native.parts) != 1 or native.parts[0] != part:
                raise ValueError('旧项目路径段不能安全转换为当前平台文件名')
            tail.append(part)
    return PROJECT.joinpath(*tail)


def relocate_project_path(value):
    mapped = _map_original_project_path(value)
    path = mapped if mapped is not None else _Path(value)
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
    return str(path) if mapped is not None else os.fspath(value)


def relocate_io_path(value):
    return relocate_project_path(value) if isinstance(value, (str, os.PathLike)) else value


class ProjectPath(type(_Path())):
    """兼容本项目历史文件位置，保留清单文本与摘要。"""
    def __new__(cls, *args):
        root = _original_project_root()
        # 保留外来平台的原始字符串，避免本机 Path 先把斜杠或盘符改写。
        mapped = _map_original_project_path(type(root)(*args)) if root is not None else None
        value = mapped if mapped is not None else _Path(*args)
        return super().__new__(cls, relocate_project_path(value))


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
