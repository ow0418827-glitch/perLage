"""只用有逐对公开视觉标签的开发集拟合外形门槛，绑定模型和预处理。"""
from src.paths import ProjectPath as Path
import math
import numpy as np
from src.visual.audit import digest,read_json,freeze_json
from src.visual.image_utils import sha256_file

CALIBRATION_POLICY='balanced_public_shape_error_midpoints_v2'


def fit_threshold(scores,labels):
    x=np.asarray(scores,dtype=np.float64)
    if x.ndim!=1 or len(x)<2 or not np.isfinite(x).all() or np.any(np.abs(x)>1.000001):raise ValueError('开发集外形分数须为有限余弦分数')
    if len(labels)!=len(x) or any(type(y) is not int or y not in (0,1) for y in labels):raise ValueError('开发标签须逐对给出整数零或一')
    if labels.count(0)!=labels.count(1) or not labels.count(0):raise ValueError('开发集须同时包含等量相似与不相似标签')
    # 加入两个端点，允许开发数据确实支持全部通过或全部拒绝。
    values=np.unique(x);thresholds=np.r_[-1.,(values[:-1]+values[1:])/2,np.nextafter(1.,2.)]
    y=np.asarray(labels)
    ranked=sorted((int(np.count_nonzero((x>=t)!=y)),float(t)) for t in thresholds)
    errors,threshold=ranked[0]
    return dict(threshold=threshold,error_count=errors,error_rate=errors/len(x),
        positive_count=labels.count(1),negative_count=labels.count(0),tie_rule='smallest_threshold',
        candidate_thresholds=[float(t) for t in thresholds])


def calibrate(manifest_path,output,backend,*,public_root):
    from src.visual.pair_selection import validate_product,unit_rows
    manifest_path=Path(manifest_path).resolve();public_root=Path(public_root).resolve()
    if not manifest_path.is_relative_to(public_root):raise ValueError('开发标签清单须位于公开数据目录')
    document=read_json(manifest_path)
    if (not {'schema_version','reviewer','pairs'}<=set(document) or set(document)-{'schema_version','reviewer','pairs','reserved_products'} or document['schema_version']!=1
            or not str(document['reviewer']).strip()):raise ValueError('开发集须记录真实公开视觉核对者与逐对标签')
    products={};pairs=[];seen=set()
    for raw in document.get('reserved_products',[]):
        p=validate_product(raw,manifest_path.parent,generation_ready=False,public_root=public_root)
        if not Path(p['image']).is_relative_to(public_root):raise ValueError('开发保留图片须位于公开目录')
        if p['id'] in products:raise ValueError('开发保留商品编号重复')
        products[p['id']]=p
    for pair in document['pairs']:
        if set(pair)!={'left','right','label','review_note'} or not str(pair['review_note']).strip():raise ValueError('每对开发图须有标签与实际核对说明')
        if type(pair['label']) is not int or pair['label'] not in (0,1):raise ValueError('开发标签只能为零或一')
        records=[]
        for field in ('left','right'):
            p=validate_product(pair[field],manifest_path.parent,generation_ready=False,public_root=public_root)
            if not Path(p['image']).is_relative_to(public_root):raise ValueError('开发图片须位于公开目录')
            if p['id'] in products and products[p['id']]['image_sha256']!=p['image_sha256']:raise ValueError('开发商品编号对应不同图片')
            products[p['id']]=p;records.append(p)
        key=tuple(sorted(p['id'] for p in records))
        if key[0]==key[1] or key in seen:raise ValueError('开发集不能包含同商品图对或重复图对')
        seen.add(key)
        pairs.append((records,pair))
    if hasattr(backend,'features_many'):backend.features_many(list(products.values()))
    rows=[]
    for records,pair in pairs:
        a,b=records
        score=float(np.clip(unit_rows(backend.features(a)['shape'])@unit_rows(backend.features(b)['shape']),-1,1))
        rows.append(dict(left_id=a['id'],right_id=b['id'],left_sha256=a['image_sha256'],right_sha256=b['image_sha256'],
            shape=score,label=pair['label'],review_note=pair['review_note']))
    fitted=fit_threshold([r['shape'] for r in rows],[r['label'] for r in rows])
    if fitted['error_rate']>=.5:raise ValueError('外形代理在平衡开发集上没有判别能力，停止校准，不输出占位门槛')
    result=dict(schema_version=2,policy=CALIBRATION_POLICY,status='calibrated',reviewer=document['reviewer'],
        development_manifest=str(manifest_path),development_manifest_sha256=sha256_file(manifest_path),
        development_product_ids=sorted(products),rows=rows,fit=fitted,
        development_image_sha256=sorted({p['image_sha256'] for p in products.values()}),
        models=backend.snapshot,models_sha256=digest(backend.snapshot),victim_memory_used=False,formal_queries_used=False)
    result['certificate_sha256']=digest(result)
    freeze_json(output,result)
    return result


def load_calibration(path,models,*,public_root=None):
    if not path:raise ValueError('缺少公开开发集校准证书；不能沿用旧阈值或自动使用占位阈值')
    if public_root is not None and not Path(path).resolve().is_relative_to(Path(public_root).resolve()):raise ValueError('校准证书须在公开目录')
    result=read_json(path);body={k:v for k,v in result.items() if k!='certificate_sha256'}
    if (result.get('policy')!=CALIBRATION_POLICY or result.get('status')!='calibrated'
            or result.get('schema_version')!=2 or result.get('certificate_sha256')!=digest(body)
            or result.get('models_sha256')!=digest(models) or result.get('models')!=models
            or result.get('victim_memory_used') is not False or result.get('formal_queries_used') is not False):
        raise ValueError('公开门槛校准证书、模型或预处理身份不一致')
    rows=result['rows']
    fitted=fit_threshold([r['shape'] for r in rows],[r['label'] for r in rows])
    ids={r[k] for r in rows for k in ('left_id','right_id')}
    if fitted!=result['fit'] or not ids<=set(result['development_product_ids']) or fitted['error_rate']>=.5:raise ValueError('校准门槛与逐对开发分数不一致或没有判别能力')
    manifest=Path(result['development_manifest'])
    if public_root is not None and not manifest.resolve().is_relative_to(Path(public_root).resolve()):raise ValueError('开发标签须在读取前确认位于公开目录')
    if sha256_file(manifest)!=result['development_manifest_sha256']:raise ValueError('公开开发标签原始记录改变')
    original=read_json(manifest)
    if original.get('reviewer')!=result['reviewer'] or len(original.get('pairs',[]))!=len(rows):raise ValueError('校准与原始公开标签数量或核对者不一致')
    originals=list(original.get('reserved_products',[]))+[p[k] for p in original['pairs'] for k in ('left','right')]
    original_ids=set();image_hashes=set()
    for p in originals:
        image=Path(p['image']);image=image if image.is_absolute() else manifest.parent/image
        if public_root is not None and not image.resolve().is_relative_to(Path(public_root).resolve()):raise ValueError('开发图片须在读取前确认位于公开目录')
        original_ids.add(p['id']);image_hashes.add(sha256_file(image))
    if sorted(original_ids)!=result['development_product_ids'] or sorted(image_hashes)!=result['development_image_sha256']:
        raise ValueError('公开开发划分的商品或图片集合改变')
    for pair,row in zip(original['pairs'],rows):
        if pair['label']!=row['label'] or pair['review_note']!=row['review_note']:raise ValueError('校准标签或核对说明改变')
        for role in ('left','right'):
            p=pair[role];image=Path(p['image']);image=image if image.is_absolute() else manifest.parent/image
            if p['id']!=row[role+'_id'] or sha256_file(image)!=row[role+'_sha256']:raise ValueError('公开开发商品身份或图片改变')
    return result
