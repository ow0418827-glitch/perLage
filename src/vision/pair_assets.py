"""将确定的图对发布为按组命名的正式输入，保留原始字节及公开选取证据。"""
import copy
from pathlib import Path
import shutil

from PIL import Image

from src.vision.audit import freeze_json, read_json, verify_model_snapshot
from src.vision.image_utils import sha256_file


def _inside(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if not path.is_relative_to(root):
        raise ValueError('正式图对及标注须位于本项目的公开数据目录')
    return path


def _copy_exact(source, destination, expected):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        if sha256_file(destination) != expected:
            raise ValueError('分组目录中的冻结图片或证据已改变：' + str(destination))
        return
    if sha256_file(source) != expected:
        raise ValueError('待发布的公开图片或证据内容改变：' + str(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if sha256_file(destination) != expected:
        raise ValueError('正式图片复制校验失败')


def _verify_choice(selection):
    if selection.get('status') != 'selected' or not selection.get('source'):
        raise ValueError('只有已经确定的合格图对才能发布')
    eligible = sorted((row for row in selection['candidates'] if row['eligible']),
                      key=lambda row: (-row['score'], row['source_id']))
    if (not eligible or eligible[0]['source_id'] != selection['source']['id']
            or selection['selected_source_id'] != selection['source']['id']
            or selection['target_id'] != selection['target']['id']):
        raise ValueError('正式图对与公开最高分选择不一致')


def load_pair(manifest_path, public_root, *, verify_models=True):
    """仅从分组目录读取正式图片及证据，不依赖早期下载缓存。"""
    manifest_path = _inside(manifest_path, public_root)
    directory = manifest_path.parent
    manifest = read_json(manifest_path)
    if (manifest.get('schema_version') != 1 or manifest.get('status') != 'selected'
            or manifest.get('policy') != 'named_public_pair_v1'):
        raise ValueError('正式图对清单版本或状态无效')
    evidence = _inside(directory / manifest['selection_record'], directory)
    if sha256_file(evidence) != manifest['selection_sha256']:
        raise ValueError('公开选取证据内容改变')
    original = read_json(evidence)
    _verify_choice(original)
    result = copy.deepcopy(original)
    for role in ('source', 'target'):
        record = copy.deepcopy(manifest[role])
        for field in ('image', 'foreground_mask'):
            if record.get(field):
                record[field] = str(_inside(directory / record[field], directory))
                if sha256_file(record[field]) != original[role][field + '_sha256']:
                    raise ValueError('分组目录中的冻结图片或前景标注改变')
        if Path(record['image']).stem != role:
            raise ValueError('正式原图和目标图必须使用约定的简单文件名')
        comparable = copy.deepcopy(record)
        for field in ('image', 'foreground_mask'):
            if original[role].get(field):
                comparable[field] = original[role][field]
        if comparable != original[role]:
            raise ValueError('商品身份、名称或公开标注与选取证据不一致')
        result[role] = record
    if verify_models:
        verify_model_snapshot(original['models'])
    result.update(pair_manifest=str(manifest_path), pair_manifest_sha256=sha256_file(manifest_path),
                  scoring_record_sha256=manifest['selection_sha256'])
    return result


def publish_pair(selection_path, directory, public_root):
    """按真实图像格式保留字节，不重新编码；冲突时停止，避免覆盖冻结输入。"""
    selection_path, directory = Path(selection_path).resolve(), _inside(directory, public_root)
    selection = read_json(selection_path)
    _verify_choice(selection)
    manifest_path = directory / 'pair.json'
    if selection.get('pair_manifest'):
        if Path(selection['pair_manifest']).resolve() != manifest_path:
            raise ValueError('已发布图对的正式目录改变，请显式使用新的运行和发布目录')
        result = load_pair(manifest_path, public_root)
        if any(result[role] != selection[role] for role in ('source', 'target')):
            raise ValueError('已发布图对与选取记录不一致')
        return result
    expected = sha256_file(selection_path)
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest['selection_sha256'] != expected:
            raise ValueError('该分组目录已有另一份冻结选择，请使用新的分组目录')
        return load_pair(manifest_path, public_root)
    manifest = dict(schema_version=1, policy='named_public_pair_v1', status='selected',
                    selection_record='metadata/selection.json', selection_sha256=expected,
                    public_selection_only=True, victim_memory_used=False)
    for role in ('source', 'target'):
        record = copy.deepcopy(selection[role])
        image = _inside(record['image'], public_root)
        with Image.open(image) as opened:
            extension = {'JPEG': '.jpg', 'PNG': '.png', 'WEBP': '.webp'}.get(opened.format)
        if extension is None:
            raise ValueError('正式图对图片格式不受支持')
        record['image'] = role + extension
        _copy_exact(image, directory / record['image'], record['image_sha256'])
        if record.get('foreground_mask'):
            mask = _inside(record['foreground_mask'], public_root)
            record['foreground_mask'] = 'metadata/' + role + '_foreground.png'
            _copy_exact(mask, directory / record['foreground_mask'], record['foreground_mask_sha256'])
        manifest[role] = record
    _copy_exact(selection_path, directory / manifest['selection_record'], expected)
    freeze_json(manifest_path, manifest)
    return load_pair(manifest_path, public_root)
