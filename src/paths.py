"""项目路径集中定义；运行不重定向到迁移来源。"""
import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / 'src'
CONFIGS = SOURCE / 'configs'
DATA = SOURCE / 'data'
OUTPUT = SOURCE / 'output'
PRIVATE = DATA / 'private'
SHARED_MEMORY = PRIVATE / 'shared_memory.json'
MODEL_ROOT = Path(os.environ.get('PERLAGE_MODEL_ROOT', PROJECT / 'models')).resolve()
VENDOR_ROOT = SOURCE / 'vendor'
ProjectPath = Path


def relocate_project_path(value):
    """保留调用接口，不再兼容或改写旧项目路径。"""
    return os.fspath(value)


def relocate_io_path(value):
    return os.fspath(value) if isinstance(value, (str, os.PathLike)) else value


def model_path(name):
    return str(MODEL_ROOT / name)
