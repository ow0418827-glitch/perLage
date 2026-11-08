"""九组配额约束的单一共享池；分组标签仅供审计，检索始终使用全部条目。"""
from collections import Counter
from contextlib import closing
import copy
import hashlib
import io
import re
import shutil
import sqlite3
import time

from PIL import Image
import requests

from src.paths import ProjectPath as Path, PUBLIC, PRIVATE, SHARED_MEMORY
from src.visual.audit import freeze_json, read_json, save_json
from src.visual.image_utils import sha256_file
from .shared_pool import load_shared_pool

GROUP_PATTERNS = {
    '01_camera': r'\bcameras?\b',
    '02_chair': r'\bchairs?\b',
    '03_cookies': r'\bcookies?\b',
    '04_curtain': r'\bcurtains?\b',
    '05_hair_extension': r'\b(hair extensions?|hair toppers?|hair buns?|ponytail extensions?)\b',
    '06_jeans': r'\bjeans?\b',
    '07_shampoo': r'\bshampoos?\b',
    '08_speaker': r'\bspeakers?\b',
    '09_toothbrush': r'\btoothbrush(es)?\b',
}
GROUP_TOKENS = ('camera','chair','cookie','curtain','hair','jean','shampoo','speaker','toothbrush')
GROUP_EXCLUSIONS = {
    '01_camera': r'accessories? bundles?|camera case|car stereo|radio receiver|camera mount|camera strap|camera bag',
    '02_chair': r'chair covers?|chair slipcovers?|chair cushions?|chair mats?|chair wheels?',
    '03_cookies': r'coffee|k-cups?|protein powder|meal replacement|shake|packaging|cookie box|cutters?|jars?|molds?|candles?',
    '04_curtain': r'lights?|curtain rods?|rod brackets?|curtain hooks?|curtain rings?|tiebacks?',
    '05_hair_extension': r'tapes?|adhesive|glue|remover|wig caps?',
    '06_jeans': r'jackets?|shorts?|chef pant',
    '07_shampoo': r'empty|refillable|dispenser|bottles set|capes?|haircut|basin|brush|shampoo cap|shampoo bowl',
    '08_speaker': r'power cord|charger|cable|adapter|stands?|cases?|covers?|mounts?|spikes',
    '09_toothbrush': r'cases?|charger|charging base|storage|toothpaste|bags?|holder|container|denture|braces|replacement head|refill|stand|organi[sz]er',
}


def relevant_public_title(group, title):
    return bool(re.search(GROUP_PATTERNS[group], title, re.I)) and not re.search(GROUP_EXCLUSIONS[group], title, re.I)


def validate_balanced_pool(path, normal_per_group=20):
    entries = load_shared_pool(path)
    expected = {(g, role): normal_per_group if role == 'normal' else 1
                for g in GROUP_PATTERNS for role in ('normal','error')}
    actual = Counter((e.get('SourceGroup'), e.get('EntryType')) for e in entries)
    if actual != expected or len(entries) != 9 * (normal_per_group + 1):
        raise ValueError('共享池必须每组二十条正常记录和一条错误记录')
    for field in ('Id', 'FinalASIN'):
        if len({e[field] for e in entries}) != len(entries):
            raise ValueError('共享池存在重复标识或重复商品：' + field)
    if len({e['PrimaryImage']['SHA256'] for e in entries}) != len(entries):
        raise ValueError('共享池存在重复图片内容')
    return entries


def _download(record, directory):
    url = record['image_url']
    stem = hashlib.sha256(url.encode('utf-8')).hexdigest()
    path = Path(directory) / 'download_cache' / (stem + '.image')
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        for attempt in range(3):
            try:
                response = requests.get(url, timeout=(10,40))
                response.raise_for_status()
                payload = response.content
                with Image.open(io.BytesIO(payload)) as image:
                    image.load()
                    if min(image.size) < 32:
                        raise ValueError('公开商品图过小')
                path.write_bytes(payload)
                break
            except (requests.RequestException, OSError):
                if attempt == 2:
                    raise
                time.sleep((1,2)[attempt])
    with Image.open(path) as image:
        image.load()
        suffix = {"JPEG":'.jpg', 'PNG':'.png', 'WEBP':'.webp'}.get(image.format)
        if suffix is None:
            raise ValueError('不支持的公开商品图格式')
    return path, suffix


def expand_pool(destination=SHARED_MEMORY, *, base_path=PRIVATE/'shared_memory.json',
                public_index=PUBLIC/'public_products_v2.sqlite', seed=20261107):
    destination = Path(destination).resolve()
    if destination.exists():
        return validate_balanced_pool(destination)
    root = destination.parent
    base = load_shared_pool(base_path)
    sources = [e for e in base if e.get('EntryType') == 'error']
    if Counter(e.get('SourceGroup') for e in sources) != Counter({g:1 for g in GROUP_PATTERNS}):
        raise ValueError('原共享池缺少九个唯一的源商品记录')
    entries, ids, hashes = [], set(), set()
    audit = dict(protocol='single_shared_pool_9x21_v1', normal_per_group=20, errors_per_group=1,
                 seed=seed, base_path=str(Path(base_path).resolve()), base_sha256=sha256_file(base_path),
                 public_index_sha256=sha256_file(public_index),
                 query_or_rank_used_for_selection=False, additions=[], exclusions=[])

    def add(row, image, suffix=None):
        digest = sha256_file(image)
        if row['FinalASIN'] in ids or digest in hashes:
            audit['exclusions'].append(dict(asin=row['FinalASIN'], reason='duplicate_product_or_image'))
            return False
        value = copy.deepcopy(row)
        suffix = suffix or Path(image).suffix.lower()
        target = root / 'images' / (digest + suffix)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(image, target)
        if sha256_file(target) != digest:
            raise ValueError('共享池图片复制失败')
        value['PrimaryImage'].update(LocalPath=target.relative_to(root).as_posix(), SHA256=digest)
        entries.append(value);ids.add(value['FinalASIN']);hashes.add(digest)
        return True

    from src.visual.visual_retrieval import resolve_image
    for row in sources:
        if not add(row, resolve_image(row, base_path)):
            raise ValueError('错误记录身份或图片重复')
    for row in base:
        if row.get('EntryType') == 'normal':
            add(row, resolve_image(row, base_path))
    counts = Counter(e['SourceGroup'] for e in entries if e['EntryType'] == 'normal')
    with closing(sqlite3.connect(Path(public_index).as_uri()+'?mode=ro',uri=True)) as db:
        for (group,pattern),token in zip(GROUP_PATTERNS.items(),GROUP_TOKENS):
            # 只按公开名称确定语义相关性，用预先固定散列次序补齐，不调用模型或读取检索名次。
            candidates=[]
            for row in db.execute('SELECT id,title,image_url,source_url FROM products WHERE lower(title) LIKE ?',('%'+token+'%',)):
                if row[0] in ids or not relevant_public_title(group,row[1]):
                    continue
                order=hashlib.sha256(f'{seed}|{group}|{row[0]}'.encode('utf-8')).hexdigest()
                candidates.append((order,dict(zip(('id','title','image_url','source_url'),row))))
            candidates.sort(key=lambda p:p[0])
            for order, record in candidates:
                if counts[group] == 20:
                    break
                try:
                    image,suffix = _download(record,root)
                except (requests.RequestException,ValueError,OSError) as error:
                    audit['exclusions'].append(dict(asin=record['id'],reason='image_unavailable',error_type=type(error).__name__))
                    continue
                row=dict(Id='public_'+hashlib.sha256((group+'|'+record['id']).encode('utf-8')).hexdigest(),
                    FinalASIN=record['id'], SourceGroup=group, EntryType='normal',
                    PrimaryImage=dict(ASIN=record['id'],Title=record['title'],SourceURL=record['image_url']),
                    RecordKind='simulated_gallery', RecordOrigin='public_catalog_reference',
                    ObservedPurchase=False, Completed=False,
                    ReferenceNote='公开商品参考；未观察到购买，没有虚构动作、价格或购买历史。',
                    PublicSource=dict(url=record['source_url'],selection_order_sha256=order))
                if add(row,image,suffix):
                    counts[group]+=1
                    audit['additions'].append(dict(group=group,asin=record['id'],title=record['title'],
                                                  image_sha256=sha256_file(image),selection_order_sha256=order))
                save_json(root/'BUILD_PROGRESS.json',dict(normal_counts=dict(counts),entries=len(entries)))
            if counts[group] != 20:
                save_json(root/'BUILD_FAILURE.json',audit)
                raise ValueError('公开候选不足以补齐本组配额：'+group)
            print(f'共享池 {group}：正常记录 {counts[group]}/20，错误记录 1/1',flush=True)
    entries.sort(key=lambda e:(e['SourceGroup'],e['EntryType']!='normal',e['Id']))
    pool=dict(schema_version=1,scope='shared',entries=entries,protocol=audit['protocol'],
              group_counts={g:dict(normal=20,error=1) for g in GROUP_PATTERNS},
              selection_seed=seed,training_usage='forbidden',
              record_kind_counts=dict(Counter(e['RecordKind'] for e in entries)))
    freeze_json(destination,pool)
    freeze_json(root/'BUILD_AUDIT.json',audit)
    validate_balanced_pool(destination)
    return entries
