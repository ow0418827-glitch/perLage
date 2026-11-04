"""统一的内容摘要、冻结和模型文件身份。"""
import hashlib
import json
from pathlib import Path
import uuid


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                   separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def freeze_json(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != json.loads(json.dumps(value)):
            raise ValueError('冻结记录改变；请创建新的运行目录：' + str(path))
        return
    save_json(path, value)


def model_files(root):
    from src.vision.image_utils import sha256_file
    root = Path(root).resolve()
    files = {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(root.rglob('*'))
             if p.is_file() and p.suffix in ('.json', '.safetensors', '.bin', '.txt', '.model')
             and p.name not in ('download_manifest.json', 'download_provenance.json')}
    if not files or not any(n.endswith(('.safetensors', '.bin')) for n in files):
        raise ValueError('本地模型缺少可核验权重：' + str(root))
    return dict(path=str(root), files=files)


def verify_model_snapshot(snapshot):
    """复用公开选取记录前重新核对模型文件，避免只按路径名接受新权重。"""
    from src.vision.image_utils import sha256_file
    def walk(value):
        if isinstance(value, dict):
            if 'path' in value and isinstance(value.get('files'), dict):
                for name, expected in value['files'].items():
                    if isinstance(expected, str) and sha256_file(Path(value['path']) / name) != expected:
                        raise ValueError('公开代理的模型文件已改变：' + name)
            for stem in ('weight', 'backbone'):
                if value.get(stem + '_path') and value.get(stem + '_sha256'):
                    if sha256_file(value[stem + '_path']) != value[stem + '_sha256']:
                        raise ValueError('公开代理权重内容改变')
            for child in value.values():
                walk(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                walk(child)
    walk(snapshot)
