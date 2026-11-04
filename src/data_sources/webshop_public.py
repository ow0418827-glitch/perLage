"""从原始商城商品文件提取公开字段，构建固定目标的源商品候选。"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
import math
from pathlib import Path
import re
from urllib.parse import urlparse

import numpy as np
from PIL import Image

from src.vision.audit import digest, freeze_json, read_json, save_json, model_files
from src.vision.image_utils import sha256_file
from src.vision.pair_selection import SelectionConfig, title_key

IMAGE_HOSTS = frozenset(('m.media-amazon.com', 'images-na.ssl-images-amazon.com',
    'images-eu.ssl-images-amazon.com', 'images.amazon.com', 'ecx.images-amazon.com'))
OWNERSHIP_POLICY = 'simulated_controlled_public_subset_sha256_even_v1'


def public_projection(item):
    """只投影编号、公开名称和主图；不提取类目树、用户指令、价格等字段。"""
    asin, name, images = item.get('asin'), item.get('name'), item.get('images')
    if (not isinstance(asin, str) or not re.fullmatch(r'[A-Za-z0-9]{10}', asin)
            or not isinstance(name, str) or not name.strip()
            or not isinstance(images, list) or not images or not isinstance(images[0], str)):
        return None
    parsed = urlparse(images[0])
    if parsed.scheme not in ('http', 'https') or parsed.hostname not in IMAGE_HOSTS or parsed.username or parsed.password:
        return None
    return dict(id=asin, title=' '.join(name.split()), image_url=images[0],
                source_url='https://www.amazon.com/dp/' + asin)


def simulated_controlled(product_id, seed, target_ids):
    if product_id in target_ids:
        return False
    value = hashlib.sha256(f'{OWNERSHIP_POLICY}|{seed}|{product_id}'.encode('utf-8')).digest()
    return int.from_bytes(value[:8], 'big') % 2 == 0


def extract_public_candidates(products_path, groups, output, *, seed=42, lexical_limit=2000):
    """有界内存流式读取；按固定公开标题规则和固定模拟可控划分提取。"""
    import ijson
    output, products_path = Path(output), Path(products_path).resolve()
    if not products_path.is_file():
        raise FileNotFoundError(products_path)
    target_ids = {g['target_id'] for g in groups}
    if len(target_ids) != len(groups) or len({g['key'] for g in groups}) != len(groups):
        raise ValueError('目标商品及任务标识须各自唯一')
    identity = dict(products_path=str(products_path), products_sha256=sha256_file(products_path),
                    groups=groups, seed=seed, lexical_limit=lexical_limit,
                    ownership_policy=OWNERSHIP_POLICY, code_sha256=sha256_file(__file__))
    final = output / 'public_extraction.json'
    if final.exists():
        saved = read_json(final)
        if saved['identity'] != identity:
            raise ValueError('公开数据或提取规则改变，请使用新的准备目录')
        return saved
    patterns = {g['key']: re.compile(g['title_pattern'], re.IGNORECASE) for g in groups}
    collected = {g['key']: {} for g in groups}
    counts = {g['key']: dict(title_matches=0, simulated_controlled_matches=0) for g in groups}
    targets, scanned, projected = {}, 0, 0
    with products_path.open('rb') as stream:
        for raw in ijson.items(stream, 'item', use_float=True):
            scanned += 1
            item = public_projection(raw)
            if item is None:
                continue
            projected += 1
            pid = item['id']
            if pid in target_ids:
                if pid in targets and targets[pid] != item:
                    raise ValueError('同一个公开目标编号存在不同资料')
                targets[pid] = item
            for group in groups:
                key = group['key']
                if not patterns[key].search(item['title']):
                    continue
                counts[key]['title_matches'] += 1
                if not simulated_controlled(pid, seed, target_ids):
                    continue
                counts[key]['simulated_controlled_matches'] += 1
                collected[key].setdefault(pid, item)
            if scanned % 200000 == 0:
                print(f'已只读扫描 {scanned} 条公开商品', flush=True)
    if set(targets) != target_ids:
        raise ValueError('原始商品文件缺少固定目标：' + ','.join(sorted(target_ids-set(targets))))
    result_groups = []
    for group in groups:
        key = group['key']
        # 超过上限时按公开编号的固定散列截取，避免文件顺序成为隐含优选依据。
        rows = sorted(collected[key].values(), key=lambda p: (digest([seed, p['id']]), p['id']))[:lexical_limit]
        result_groups.append(dict(key=key, target=targets[group['target_id']], candidates=rows,
                                  counts=dict(counts[key], retained_lexical_candidates=len(rows))))
    result = dict(schema_version=1, identity=identity, scanned_products=scanned,
                  usable_public_products=projected, projected_fields=['asin', 'name', 'images[0]'],
                  excluded_fields=['category_tree', 'attributes', 'human_instructions', 'goals',
                                   'history', 'prices', 'private_rankings'],
                  ownership_is_experimental_assumption=True, groups=result_groups)
    freeze_json(final, result)
    return result


def load_text_proxy(model_name, device='cuda'):
    from sentence_transformers import SentenceTransformer
    from huggingface_hub import snapshot_download
    root = Path(model_name)
    if not root.is_dir():
        root = Path(snapshot_download(model_name, local_files_only=True,
                    allow_patterns=['*.json', '*.txt', 'model.safetensors', '1_Pooling/*']))
    model = SentenceTransformer(str(root.resolve()), local_files_only=True, device=device)
    return model, model_files(root)


def semantic_shortlist(extracted, output, *, limit=30, model_name='sentence-transformers/all-MiniLM-L6-v2',
                       device='cuda', model=None, model_snapshot=None):
    """在图片下载之前以同一公开语义门槛粗筛，最终四项评分仍完整执行。"""
    output = Path(output)
    config = SelectionConfig()
    identity = dict(extraction_sha256=digest(extracted), candidate_limit=limit,
                    semantic_config=dict(min=config.semantic_min, max=config.semantic_max,
                                         center=config.semantic_center, sigma=config.semantic_sigma),
                    model_name=model_name, code_sha256=sha256_file(__file__))
    final = output / 'semantic_shortlist.json'
    if final.exists():
        saved = read_json(final)
        if saved['identity'] != identity:
            raise ValueError('公开语义粗筛输入或实现改变')
        return saved
    if model is None:
        model, model_snapshot = load_text_proxy(model_name, device)
    results = []
    for group in extracted['groups']:
        target = group['target']
        records = group['candidates']
        if not records:
            raise ValueError('公开词法范围没有模拟可控候选：' + group['key'])
        vectors = np.asarray(model.encode([target['title']] + [r['title'] for r in records],
                        batch_size=64, normalize_embeddings=True, show_progress_bar=False), dtype=np.float64)
        similarities = np.clip(vectors[1:] @ vectors[0], -1, 1)
        scored = []
        for product, cosine in zip(records, similarities):
            duplicate = title_key(product['title']) == title_key(target['title'])
            eligible = not duplicate and config.semantic_min <= cosine <= config.semantic_max
            score = math.exp(-0.5*((float(cosine)-config.semantic_center)/config.semantic_sigma)**2)
            scored.append(dict(product=product, cosine=float(cosine), window_score=score,
                               eligible=bool(eligible), same_title=duplicate))
        retained = sorted((r for r in scored if r['eligible']), key=lambda r: (-r['window_score'], r['product']['id']))[:limit]
        results.append(dict(key=group['key'], target=target,
                            candidates=[dict(r['product'], coarse_semantic_cosine=r['cosine']) for r in retained],
                            coarse_scores=scored, counts=dict(group['counts'], semantic_eligible=sum(r['eligible'] for r in scored),
                                                             retained_for_images=len(retained))))
        print(f"{group['key']}：公开语义合格 {sum(r['eligible'] for r in scored)}，固定保留 {len(retained)} 个图片候选", flush=True)
    result = dict(schema_version=1, identity=identity, text_model=model_snapshot,
                  text_max_sequence_length=getattr(model, 'max_seq_length', None), groups=results)
    freeze_json(final, result)
    return result


def download_public_image(product, output, *, fetcher=None):
    """固定取公开主图，有限重试；仅使用当前任务自己的图片缓存。"""
    import requests
    output = Path(output)
    record_file = output / 'image_records' / (product['id'] + '.json')
    if record_file.exists():
        prior = read_json(record_file)
        if prior['image_url'] != product['image_url']:
            raise ValueError('公开主图地址改变')
        path = Path(prior['image'])
        if not path.is_file() or sha256_file(path) != prior['image_sha256']:
            raise ValueError('公开原图缓存缺失或内容改变')
        return prior
    parsed = urlparse(product['image_url'])
    if parsed.scheme not in ('http', 'https') or parsed.hostname not in IMAGE_HOSTS:
        raise ValueError('只接受公开商品图像主机')
    last_error = None
    for attempt in range(2):
        try:
            if fetcher is None:
                response = requests.get(product['image_url'], timeout=(10, 30), stream=True)
                response.raise_for_status()
                if urlparse(response.url).hostname not in IMAGE_HOSTS:
                    raise ValueError('图片下载重定向到未知主机')
                chunks, size = [], 0
                for chunk in response.iter_content(128*1024):
                    size += len(chunk)
                    if size > 12*1024*1024:
                        raise ValueError('公开图片超过十二兆字节上限')
                    chunks.append(chunk)
                response.close()
                body = b''.join(chunks)
            else:
                body = fetcher(product['image_url'])
            with Image.open(io.BytesIO(body)) as image:
                extension = {'JPEG': '.jpg', 'PNG': '.png', 'WEBP': '.webp'}.get(image.format)
                if not extension or min(image.size) < 64 or image.width*image.height > 25000000:
                    raise ValueError('公开图片格式或尺寸不合适')
                width, height = image.size
                image.verify()
            image_hash = hashlib.sha256(body).hexdigest()
            path = output / 'images' / (image_hash + extension)
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and sha256_file(path) != image_hash:
                raise ValueError('同名图片缓存内容不符')
            if not path.exists():
                path.write_bytes(body)
            record = dict(id=product['id'], title=product['title'], source_url=product['source_url'],
                          image_url=product['image_url'], image=str(path.resolve()), image_sha256=image_hash,
                          width=width, height=height, image_index=0, attempts=attempt+1)
            freeze_json(record_file, record)
            return record
        except Exception as error:
            last_error = error
    raise RuntimeError(f"公开主图下载失败 {product['id']}：{last_error}")


def download_shortlist(shortlist, output, *, workers=6):
    output = Path(output)
    final = output / 'download_manifest.json'
    products = {p['id']: p for group in shortlist['groups'] for p in [group['target']]+group['candidates']}
    records, failures = {}, {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(download_public_image, p, output): pid for pid, p in products.items()}
        for future in as_completed(jobs):
            pid = jobs[future]
            try:
                records[pid] = future.result()
            except Exception as error:
                failures[pid] = dict(type=type(error).__name__, reason=str(error))
    result = dict(shortlist_sha256=digest(shortlist), records=records, failures=failures,
                  downloads=len(records), failures_count=len(failures))
    freeze_json(final, result)
    for group in shortlist['groups']:
        if group['target']['id'] not in records:
            raise ValueError('固定公开目标图未下载成功：'+group['key'])
    print(f'公开原图下载完成：成功 {len(records)}，失败 {len(failures)}', flush=True)
    return result
