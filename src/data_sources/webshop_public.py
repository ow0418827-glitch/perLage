"""公开字段索引、粗类型候选抽样及有限主图下载；不计算标题相似度。"""
from concurrent.futures import ThreadPoolExecutor,as_completed
from contextlib import closing
import hashlib
import heapq
import io
import json
from pathlib import Path
import re
import sqlite3
from urllib.parse import urlparse
from PIL import Image
from src.vision.audit import digest,freeze_json,read_json
from src.vision.image_utils import sha256_file
from src.vision.product_types import classify_public_type,type_compatibility,TYPE_RULES_SHA256

IMAGE_HOSTS=frozenset(('m.media-amazon.com','images-na.ssl-images-amazon.com','images-eu.ssl-images-amazon.com','images.amazon.com','ecx.images-amazon.com'))
OWNERSHIP_POLICY='simulated_controlled_public_subset_sha256_even_v1'
INDEX_POLICY='public_projection_and_derived_coarse_type_index_v2'


def public_projection(item):
    asin,name,images=item.get('asin'),item.get('name'),item.get('images')
    if (not isinstance(asin,str) or not re.fullmatch(r'[A-Za-z0-9]{10}',asin)
            or not isinstance(name,str) or not name.strip() or not isinstance(images,list)
            or not images or not isinstance(images[0],str)):return None
    parsed=urlparse(images[0])
    if parsed.scheme not in ('http','https') or parsed.hostname not in IMAGE_HOSTS or parsed.username or parsed.password:return None
    return dict(id=asin,title=' '.join(name.split()),image_url=images[0],source_url='https://www.amazon.com/dp/'+asin)


def simulated_controlled(product_id,seed,target_ids):
    if product_id in target_ids:return False
    value=hashlib.sha256(f'{OWNERSHIP_POLICY}|{seed}|{product_id}'.encode('utf-8')).digest()
    return int.from_bytes(value[:8],'big')%2==0


def build_public_index(products_path,index_path,*,verify_source=False):
    """首次流式扫描；复用时按源路径、字节数、修改时间和类型规则检查，显式选项可复核原文件摘要。"""
    import ijson
    source=Path(products_path).resolve();index=Path(index_path).resolve();stat=source.stat()
    identity=dict(policy=INDEX_POLICY,source_path=str(source),source_bytes=stat.st_size,source_mtime_ns=stat.st_mtime_ns,type_rules_sha256=TYPE_RULES_SHA256)
    if index.exists():
        with closing(sqlite3.connect(index.as_uri()+'?mode=ro',uri=True)) as db:
            saved=json.loads(db.execute('SELECT value FROM meta WHERE key=?',('manifest',)).fetchone()[0])
        if saved['identity']!=identity:raise ValueError('公开源文件或类型规则改变，请重建独立索引')
        if verify_source and sha256_file(source)!=saved['source_sha256']:raise ValueError('原始商品文件内容改变')
        return saved
    index.parent.mkdir(parents=True,exist_ok=True);temporary=index.with_name(index.name+'.building')
    if temporary.exists():raise ValueError('发现未完成公开索引，请使用新索引路径或核查失败证据')
    class HashedReader:
        def __init__(self,stream):self.stream=stream;self.hash=hashlib.sha256()
        def read(self,n=-1):
            data=self.stream.read(n);self.hash.update(data);return data
    scanned=projected=0
    with closing(sqlite3.connect(temporary)) as db:
        db.execute('CREATE TABLE products(id TEXT PRIMARY KEY,title TEXT,image_url TEXT,source_url TEXT,category TEXT)')
        db.execute('CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT)')
        with source.open('rb') as stream:
            reader=HashedReader(stream)
            for raw in ijson.items(reader,'item',use_float=True):
                scanned+=1;p=public_projection(raw)
                if p is None:continue
                projected+=1;category=classify_public_type(p)['category']
                prior=db.execute('SELECT title,image_url,source_url FROM products WHERE id=?',(p['id'],)).fetchone()
                row=(p['title'],p['image_url'],p['source_url'])
                if prior is not None and prior!=row:raise ValueError('同一公开商品编号存在不同字段')
                db.execute('INSERT OR IGNORE INTO products VALUES(?,?,?,?,?)',(p['id'],*row,category))
                if scanned%200000==0:print(f'已只读建立 {scanned} 条公开商品索引',flush=True)
            # 某些解析器在数组结束后没有继续读取尾部空白。
            while reader.read(1024*1024):pass
            source_hash=reader.hash.hexdigest()
        db.execute('CREATE INDEX public_category ON products(category)')
        result=dict(identity=identity,source_sha256=source_hash,scanned_products=scanned,usable_public_products=projected,
            stored_fields=['id','title','image_url','source_url','category_derived_from_public_title'],private_fields_read=False)
        db.execute('INSERT INTO meta VALUES(?,?)',('manifest',json.dumps(result,ensure_ascii=False)))
        db.commit()
    temporary.replace(index)
    return result


def extract_public_candidates(products_path,groups,output,*,seed=42,lexical_limit=32,index_path=None,excluded_ids=(),type_reviews=None,verify_source=False):
    """从可复用公开索引均匀截取有限候选；参数旧名仅表示元数据候选预算。"""
    output=Path(output);source=Path(products_path).resolve();index=Path(index_path or output.parent/'public_products_v2.sqlite')
    manifest=build_public_index(source,index,verify_source=verify_source)
    target_ids={g['target_id'] for g in groups};excluded=set(excluded_ids);type_reviews=type_reviews or {}
    if len(target_ids)!=len(groups) or len({g['key'] for g in groups})!=len(groups):raise ValueError('目标及组名须唯一')
    if target_ids&excluded:raise ValueError('正式目标与开发校准商品重叠')
    identity=dict(index_manifest_sha256=digest(manifest),groups=groups,seed=seed,candidate_limit=lexical_limit,
        ownership_policy=OWNERSHIP_POLICY,excluded_ids=sorted(excluded),type_reviews=type_reviews)
    final=output/'public_extraction.json'
    if final.exists():
        saved=read_json(final)
        if saved['identity']!=identity:raise ValueError('公开候选规则改变，请使用新目录')
        return saved
    results=[]
    with closing(sqlite3.connect(index.resolve().as_uri()+'?mode=ro',uri=True)) as db:
        def product(row):
            p=dict(zip(('id','title','image_url','source_url'),row[:4]))
            if p['id'] in type_reviews:p['type_review']=type_reviews[p['id']]
            return p
        for group in groups:
            row=db.execute('SELECT id,title,image_url,source_url FROM products WHERE id=?',(group['target_id'],)).fetchone()
            if row is None:raise ValueError('公开索引缺少固定目标：'+group['target_id'])
            target=product(row);category=classify_public_type(target)
            if category['category'] is None:raise ValueError('目标公开类型未解决，须提供逐商品类型核对：'+target['id'])
            # 类别索引缩小范围；公开人工核对可显式纠正原有未知/多义条目。
            from src.vision.product_types import RULES,COMPATIBLE
            cats=[c for c in RULES if c==category['category'] or frozenset((c,category['category'])) in COMPATIBLE]
            placeholders=','.join('?' for _ in cats)
            def matching():
                seen=set()
                for record in db.execute(f'SELECT id,title,image_url,source_url FROM products WHERE category IN ({placeholders})',cats):
                    p=product(record);seen.add(p['id']);yield p
                for pid in sorted(type_reviews):
                    if pid not in seen:
                        r=db.execute('SELECT id,title,image_url,source_url FROM products WHERE id=?',(pid,)).fetchone()
                        if r:yield product(r)
            counts=dict(type_compatible=0,simulated_controlled=0,development_excluded=0)
            def eligible():
                for p in matching():
                    if not type_compatibility(p,target)['compatible']:continue
                    counts['type_compatible']+=1
                    if p['id'] in excluded:counts['development_excluded']+=1;continue
                    if not simulated_controlled(p['id'],seed,target_ids):continue
                    counts['simulated_controlled']+=1
                    yield p
            retained=heapq.nsmallest(lexical_limit,eligible(),key=lambda p:(digest(['public_candidate_uniform_v2',seed,target['id'],p['id']]),p['id']))
            results.append(dict(key=group['key'],target=target,candidates=retained,counts=dict(counts,retained_for_images=len(retained))))
    result=dict(schema_version=2,identity=identity,index_path=str(index.resolve()),index_manifest=manifest,
        scanned_products=manifest['scanned_products'],usable_public_products=manifest['usable_public_products'],
        groups=results,projected_fields=['asin','name','images[0]'],victim_memory_used=False,formal_queries_used=False,
        ownership_is_experimental_assumption=True)
    freeze_json(final,result);return result


def random_shortlist(extracted,output):
    """候选已按固定公开种子截取；保留全部，不加载文本模型或按语义排序。"""
    result=dict(schema_version=2,policy='coarse_type_and_public_hash_sample_v2',extraction_sha256=digest(extracted),
        groups=extracted['groups'],text_model_used=False,visual_score_used=False)
    freeze_json(Path(output)/'candidate_shortlist.json',result);return result


def download_public_image(product,output,*,fetcher=None):
    import requests
    output=Path(output);record_file=output/'image_records'/(product['id']+'.json')
    if record_file.exists():
        prior=read_json(record_file)
        if prior['image_url']!=product['image_url'] or sha256_file(prior['image'])!=prior['image_sha256']:raise ValueError('公开主图地址或缓存内容改变')
        return prior
    parsed=urlparse(product['image_url'])
    if parsed.scheme not in ('http','https') or parsed.hostname not in IMAGE_HOSTS:raise ValueError('只接受公开商品图像主机')
    for attempt in range(2):
        try:
            if fetcher is None:
                with requests.get(product['image_url'],timeout=(10,30),stream=True) as response:
                    response.raise_for_status()
                    if urlparse(response.url).hostname not in IMAGE_HOSTS:raise ValueError('公开图片重定向到未知主机')
                    body=b''
                    for part in response.iter_content(128*1024):
                        body+=part
                        if len(body)>12*1024*1024:raise ValueError('图片超过十二兆字节上限')
            else:body=fetcher(product['image_url'])
            with Image.open(io.BytesIO(body)) as image:
                extension={'JPEG':'.jpg','PNG':'.png','WEBP':'.webp'}.get(image.format)
                if not extension or min(image.size)<64 or image.width*image.height>25000000:raise ValueError('图片格式或尺寸不合适')
                width,height=image.size;image.verify()
            image_hash=hashlib.sha256(body).hexdigest();path=output/'images'/(image_hash+extension);path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists() and sha256_file(path)!=image_hash:raise ValueError('同名图片缓存内容不符')
            if not path.exists():path.write_bytes(body)
            record=dict(id=product['id'],title=product['title'],source_url=product['source_url'],image_url=product['image_url'],
                image=str(path.resolve()),image_sha256=image_hash,width=width,height=height,image_index=0,attempts=attempt+1)
            freeze_json(record_file,record);return record
        except Exception as error:last=error
    raise RuntimeError(f'公开主图下载失败 {product["id"]}：{last}')


def download_shortlist(shortlist,output,*,workers=6):
    output=Path(output);products={p['id']:p for g in shortlist['groups'] for p in [g['target']]+g['candidates']};records={};failures={}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs={pool.submit(download_public_image,p,output):pid for pid,p in products.items()}
        for future in as_completed(jobs):
            pid=jobs[future]
            try:records[pid]=future.result()
            except Exception as error:failures[pid]=dict(type=type(error).__name__,reason=str(error))
    result=dict(shortlist_sha256=digest(shortlist),records=records,failures=failures,downloads=len(records),failures_count=len(failures))
    freeze_json(output/'download_manifest.json',result)
    for g in shortlist['groups']:
        if g['target']['id'] not in records:raise ValueError('固定目标主图下载失败：'+g['key'])
    return result
