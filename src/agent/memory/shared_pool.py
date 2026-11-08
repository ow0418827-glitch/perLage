"""唯一的共享长期记忆池；不提供按商品组构建或过滤记忆的入口。"""
import copy
from src.paths import ProjectPath as Path
import shutil

from src.paths import SHARED_MEMORY
from src.visual.audit import digest, freeze_json, read_json
from src.visual.image_utils import sha256_file
from src.visual.visual_retrieval import memory_index_text, resolve_image


def load_shared_pool(path=SHARED_MEMORY):
    path = Path(path).resolve()
    pool = read_json(path)
    if (not isinstance(pool, dict) or pool.get('schema_version') != 1
            or pool.get('scope') != 'shared' or not isinstance(pool.get('entries'), list)
            or not pool['entries'] or 'groups' in pool):
        raise ValueError('必须使用统一共享池；不接受按组记忆或未经共享池工具登记的旧列表')
    entries = pool['entries']
    ids = [entry['Id'] for entry in entries]
    if any(not isinstance(i, str) or not i for i in ids) or len(ids) != len(set(ids)):
        raise ValueError('共享池必须有全局唯一的非空记忆标识')
    for entry in entries:
        memory_index_text(entry)
        if entry.get('RetrievalImage') is not None or entry.get('ImageSubstitution'):
            raise ValueError('基础共享池不得携带旧对抗图片或替换记录')
        if entry.get('RecordKind') not in ('purchase_memory', 'simulated_gallery'):
            raise ValueError('必须区分真实购买记忆与模拟图库')
        if entry['RecordKind'] == 'purchase_memory' and entry.get('Completed') is not True:
            raise ValueError('真实购买记忆需要完成证据')
        if entry['RecordKind'] == 'simulated_gallery' and entry.get('Completed') is True:
            raise ValueError('模拟图库不能标记为真实购买')
        image = resolve_image(entry, path)
        if image is None or entry['PrimaryImage'].get('SHA256') != sha256_file(image):
            raise ValueError('共享池图片必须具备可核验的内容摘要')
    return entries


def build_shared_pool(inputs, destination=SHARED_MEMORY, *, kind):
    """显式导入输入数据；保留完整历史，把图片复制到新项目独立目录。"""
    if kind not in ('purchase_memory', 'simulated_gallery', 'preserve'):
        raise ValueError('须明确声明输入是真实购买记忆还是模拟图库')
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('共享基础库已存在；请另建输入库，避免覆盖：' + str(destination))
    entries, seen, sources = [], {}, []
    for given in inputs:
        given = Path(given).resolve()
        document = read_json(given)
        rows = document.get('entries') if isinstance(document, dict) else document
        if not isinstance(rows, list):
            raise ValueError('输入必须为记忆列表或共享池对象')
        sources.append(dict(path=str(given), sha256=sha256_file(given)))
        for row in rows:
            if row.get('RetrievalImage') or row.get('ImageSubstitution'):
                raise ValueError('只允许导入未替换图片的基础记忆')
            memory_index_text(row)
            image = resolve_image(row, given)
            if image is None:
                raise ValueError('共享池不能缺少商品图片')
            value = copy.deepcopy(row)
            entry_kind = row.get('RecordKind') if kind == 'preserve' else kind
            if entry_kind not in ('purchase_memory', 'simulated_gallery'):
                raise ValueError('保留来源导入要求每条输入已有明确的真实购买或模拟标签')
            value['RecordKind'] = entry_kind
            if entry_kind == 'purchase_memory' and value.get('Completed') is not True:
                raise ValueError('不能把未完成记录当购买记忆')
            if entry_kind == 'simulated_gallery':
                if value.get('Completed') is True:
                    raise ValueError('请为模拟图库提供未标记真实购买的输入')
                value['Completed'] = False
            image_hash = sha256_file(image)
            normalized = copy.deepcopy(value)
            normalized['PrimaryImage'].pop('LocalPath', None)
            normalized['PrimaryImage']['SHA256'] = image_hash
            identity = digest(normalized)
            if identity in seen:
                continue
            seen[identity] = True
            value['OriginalId'] = str(value.get('Id', ''))
            value['Id'] = 'shared_' + identity
            target = destination.parent / 'images' / (image_hash + image.suffix.lower())
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copyfile(image, target)
            if sha256_file(target) != image_hash:
                raise ValueError('共享池图片复制摘要不符')
            value['PrimaryImage'].update(LocalPath=target.relative_to(destination.parent).as_posix(), SHA256=image_hash)
            entries.append(value)
    if not entries:
        raise ValueError('共享池不能是空库')
    pool = dict(schema_version=1, scope='shared', entries=entries, input_sources=sources)
    freeze_json(destination, pool)
    load_shared_pool(destination)
    return pool


def evaluation_banks(entries, pool_path, selected_source, frozen_image, *, source_memory_id=None):
    """评估端实验桩：只在副本替换一张图片，基础共享库保持只读。"""
    eligible = [e for e in entries if e['FinalASIN'] == selected_source['id']]
    if source_memory_id is not None:
        eligible = [e for e in eligible if e['Id'] == source_memory_id]
    if len(eligible) != 1:
        raise ValueError('共享池中须唯一绑定所选源商品；存在多次购买时由评估端指定记忆标识')
    chosen = eligible[0]
    if chosen['PrimaryImage']['SHA256'] != selected_source['image_sha256']:
        raise ValueError('待模拟记忆的原图与公开所选源图不符')
    if memory_index_text(chosen) != selected_source['title']:
        raise ValueError('待模拟记忆的实际商品名与公开所选源商品不符')
    clean, modified = copy.deepcopy(entries), copy.deepcopy(entries)
    for bank in (clean, modified):
        for entry in bank:
            entry['PrimaryImage']['LocalPath'] = str(resolve_image(entry, pool_path))
    replacement = next(e for e in modified if e['Id'] == chosen['Id'])
    replacement['RetrievalImage'] = dict(ASIN=chosen['FinalASIN'], LocalPath=str(Path(frozen_image).resolve()),
                                       SHA256=sha256_file(frozen_image))
    # 原始历史及商品名称保持逐字段一致，替换权限属于评估端。
    if [{k: v for k, v in e.items() if k != 'RetrievalImage'} for e in modified] != clean:
        raise ValueError('图片替换意外改变了历史内容')
    return {'clean': clean, 'modified': modified}, chosen['Id']
