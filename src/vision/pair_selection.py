"""公开图对：类型兼容、灰度外形门槛、去重与固定种子均匀抽样。"""
from dataclasses import asdict,dataclass
import hashlib
import math
from pathlib import Path
import re
import unicodedata
import numpy as np
from PIL import Image

from src.vision.audit import digest
from src.vision.image_utils import sha256_file
from src.vision.product_types import type_compatibility,TYPE_RULES_SHA256
from src.vision.shape_features import LocalShapeBackend as LocalSelectionBackend

SELECTION_POLICY='public_shape_gate_uniform_v2'
DRAW_POLICY='sha256_rejection_uniform_index_v1'


@dataclass(frozen=True)
class SelectionConfig:
    seed:int=42
    calibration_path:str|None=None
    min_image_side:int=64

    def validate(self):
        if type(self.seed) is not int or type(self.min_image_side) is not int or self.min_image_side<16:
            raise ValueError('选图种子须为整数，最短边门槛须为不小于十六的整数')
        if not isinstance(self.calibration_path,str) or not self.calibration_path.strip():
            raise ValueError('新选图流程须提供真实公开开发集校准证书，不提供默认外形阈值')


def unit_rows(value):
    a=np.asarray(value,dtype=np.float64)
    if a.ndim not in (1,2) or not a.size or not np.isfinite(a).all():raise ValueError('选取特征必须为有限非空向量')
    norm=np.linalg.norm(a,axis=-1,keepdims=True)
    if np.any(norm<=1e-12):raise ValueError('选取特征不能是零向量')
    return a/norm


def title_key(title):
    return re.sub(r'[\W_]+','',unicodedata.normalize('NFKC',title).casefold())


def uniform_draw(eligible_ids,target_id,seed):
    ids=sorted(eligible_ids)
    if not ids or len(ids)!=len(set(ids)):raise ValueError('抽样集合须非空且编号唯一')
    context=digest(dict(policy=DRAW_POLICY,target_id=target_id,seed=seed,eligible_ids=ids))
    bound=2**256-(2**256%len(ids));counter=0
    while True:
        value=int.from_bytes(hashlib.sha256(f'{context}|{counter}'.encode('utf-8')).digest(),'big')
        if value<bound:break
        counter+=1
    index=value%len(ids)
    return dict(policy=DRAW_POLICY,seed=seed,eligible_ids=ids,index=index,selected_source_id=ids[index],counter=counter)


def image_quality(product,minimum):
    try:
        if sha256_file(product['image'])!=product['image_sha256']:return dict(accepted=False,reason='image_digest_changed')
        with Image.open(product['image']) as opened:
            width,height=opened.size;fmt=opened.format;opened.verify()
        accepted=fmt in ('JPEG','PNG','WEBP') and min(width,height)>=minimum
        return dict(accepted=accepted,width=width,height=height,format=fmt,minimum_side=minimum)
    except (OSError,ValueError):return dict(accepted=False,reason='unreadable_image')


def score_pair(source,target,source_features,target_features,*,threshold,config=SelectionConfig()):
    """只判断合格与否；外形分数不参与最终抽样排序。"""
    compatibility=type_compatibility(source,target)
    quality=image_quality(source,config.min_image_side);target_quality=image_quality(target,config.min_image_side)
    reasons=[]
    if source['id']==target['id']:reasons.append('same_product_id')
    if title_key(source['title'])==title_key(target['title']):reasons.append('same_normalized_title')
    if source['image_sha256']==target['image_sha256']:reasons.append('same_image_bytes')
    if source.get('owned') is not True:reasons.append('source_not_attacker_owned')
    if not compatibility['resolved']:reasons.append('public_type_review_required')
    elif not compatibility['compatible']:reasons.append('obvious_product_type_conflict')
    if not quality['accepted'] or not target_quality['accepted']:reasons.append('public_image_quality_failed')
    shape=None
    if source_features is not None and target_features is not None:
        shape=float(np.clip(unit_rows(source_features['shape'])@unit_rows(target_features['shape']),-1,1))
        if shape<threshold:reasons.append('shape_below_calibrated_threshold')
    elif not reasons:reasons.append('shape_not_evaluated')
    return dict(source_id=source['id'],target_id=target['id'],eligible=not reasons,rejected_reasons=reasons,
        shape_similarity=shape,type_compatibility=compatibility,image_quality=quality)


def choose_pair(target,candidates,backend,config=SelectionConfig(),*,excluded_source_ids=(),public_root=None):
    from src.vision.shape_calibration import load_calibration
    config.validate()
    certificate=load_calibration(config.calibration_path,backend.snapshot,public_root=public_root)
    ids=[p['id'] for p in candidates]
    if not ids or len(ids)!=len(set(ids)):raise ValueError('公开候选标识不能重复或为空')
    if set(ids+[target['id']])&set(certificate['development_product_ids']):
        raise ValueError('正式候选或目标与公开开发集商品重叠')
    dev_images=set(certificate['development_image_sha256'])
    if {p['image_sha256'] for p in candidates+[target]}&dev_images:raise ValueError('正式图片与公开开发集图片重复')
    if not image_quality(target,config.min_image_side)['accepted']:raise ValueError('固定目标图片质量未通过')
    if not type_compatibility(target,target)['resolved']:raise ValueError('固定目标的公开类型须先核对')
    excluded=set(excluded_source_ids);ready=[]
    for p in candidates:
        if (p['id'] not in excluded and type_compatibility(p,target)['compatible']
                and image_quality(p,config.min_image_side)['accepted']):ready.append(p)
    if hasattr(backend,'features_many'):backend.features_many([target]+ready)
    target_features=backend.features(target)
    rows=[];seen_images={}
    ready_ids={p['id'] for p in ready}
    for p in sorted(candidates,key=lambda p:p['id']):
        row=score_pair(p,target,backend.features(p) if p['id'] in ready_ids else None,target_features,
            threshold=certificate['fit']['threshold'],config=config)
        if p['id'] in excluded:row['rejected_reasons'].append('reserved_in_public_batch')
        if not row['rejected_reasons']:
            if p['image_sha256'] in seen_images:row['rejected_reasons'].append('duplicate_candidate_image')
            else:seen_images[p['image_sha256']]=p['id']
        row['eligible']=not row['rejected_reasons'];rows.append(row)
    eligible=[r['source_id'] for r in rows if r['eligible']]
    draw=uniform_draw(eligible,target['id'],config.seed) if eligible else None
    return dict(protocol=SELECTION_POLICY,status='selected' if eligible else 'no_eligible_pair',
        selected_source_id=draw['selected_source_id'] if draw else None,target_id=target['id'],candidates=rows,
        configuration=asdict(config),draw=draw,calibration=certificate,models=backend.snapshot,
        type_rules_sha256=TYPE_RULES_SHA256,excluded_source_ids=sorted(excluded),
        victim_memory_used=False,victim_queries_used=False,target_system_probed=False)


def verify_selection(selection):
    """验证新抽样；旧最高分记录仅作既有冻结图对的读取兼容。"""
    if selection.get('status')!='selected' or not selection.get('source'):raise ValueError('只有已经确定的合格图对才能发布')
    eligible=[r for r in selection['candidates'] if r['eligible']]
    if selection.get('protocol')==SELECTION_POLICY:
        draw=uniform_draw([r['source_id'] for r in eligible],selection['target_id'],selection['configuration']['seed'])
        if draw!=selection['draw']:raise ValueError('冻结抽样记录与合格集合或随机种子不一致')
        expected=draw['selected_source_id']
        certificate=selection['calibration'];body={k:v for k,v in certificate.items() if k!='certificate_sha256'}
        if digest(body)!=certificate['certificate_sha256'] or digest(selection['models'])!=certificate['models_sha256']:
            raise ValueError('冻结图对的开发门槛或模型身份改变')
        for row in eligible:
            if row['rejected_reasons'] or row['shape_similarity']<certificate['fit']['threshold']:
                raise ValueError('抽样集合中存在不合格候选')
    else:
        # 不注册旧选择入口，不重新计算旧分数，不修改已有实验。
        if selection.get('calibration')!='provisional_public_only_heuristics_not_success_probabilities':
            raise ValueError('公开选图协议未知')
        eligible=sorted(eligible,key=lambda r:(-r['score'],r['source_id']))
        expected=eligible[0]['source_id'] if eligible else None
    if (expected!=selection['selected_source_id'] or expected!=selection['source']['id']
            or selection['target_id']!=selection['target']['id']):raise ValueError('正式图对与公开选择证据不一致')


def validate_product(record,base,*,owned=False,generation_ready=True,public_root=None):
    allowed={'id','title','image','owned','source_url','source_note','boxes','text_regions','text_regions_reviewed',
        'foreground_mask','image_sha256','foreground_mask_sha256','type_review'}
    if not isinstance(record,dict) or set(record)-allowed:
        raise ValueError('公开商品记录含未知字段；禁止传入历史、私有类目或检索结果')
    product=dict(record)
    for field in ('id','title','image'):
        if not isinstance(product.get(field),str) or not product[field].strip():raise ValueError('公开商品缺少有效字段：'+field)
    from src.vision.retrieval_objective import validate_public_title
    product['title']=validate_public_title(product['title'])
    if owned and product.get('owned') is not True:raise ValueError('原商品必须明确声明为可修改商品')
    if not (product.get('source_url') or product.get('source_note')):raise ValueError('须记录公开来源或模拟可控说明')
    for field in ('image','foreground_mask'):
        if product.get(field):
            path=Path(product[field]);path=(path if path.is_absolute() else Path(base)/path).resolve()
            if public_root is not None and not path.is_relative_to(Path(public_root).resolve()):
                raise ValueError('公开图片和标注须在读取内容前确认位于公开数据目录')
            if not path.is_file():raise FileNotFoundError('公开图片或掩码不存在：'+str(path))
            actual=sha256_file(path)
            if product.get(field+'_sha256') and actual!=product[field+'_sha256']:raise ValueError('公开图片或掩码摘要改变')
            product[field]=str(path);product[field+'_sha256']=actual
    if product.get('type_review') is not None:
        from src.vision.product_types import classify_public_type
        classify_public_type(product)
    if generation_ready:
        if product.get('text_regions_reviewed') is not True or not isinstance(product.get('text_regions'),list):
            raise ValueError('选中图对须完成公开文字及品牌区域核对，再进行发布或生成')
        for rect in product['text_regions']:
            if (len(rect)!=4 or not all(math.isfinite(x) for x in rect)
                    or not (0<=rect[0]<rect[2]<=1 and 0<=rect[1]<rect[3]<=1)):raise ValueError('文字区域使用零至一矩形坐标')
        if not product.get('foreground_mask') and not product.get('boxes'):raise ValueError('选中图对须提供公开主体框或掩码')
    return product
