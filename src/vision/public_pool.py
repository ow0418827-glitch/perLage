"""独立公开竞争池与生成残差的受控研究组件；不接收受害图库。"""
from __future__ import annotations
from src.paths import relocate_io_path

from dataclasses import dataclass
import hashlib
import json
from src.paths import ProjectPath as Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PublicMarginConfig:
    margin: float = .01
    temperature: float = .01
    image_weight: float = .25
    top1_weight: float = .25


def public_thresholds(query_features, target_feature, image_features, text_features):
    """同一查询与每条公开图文键比较；前三门槛不含攻击候选。"""
    if image_features.ndim != 2 or image_features.shape != text_features.shape or len(image_features) < 3:
        raise ValueError('公开竞争者至少三条，图文维度必须一致')
    values = (query_features, target_feature, image_features, text_features)
    if any(not torch.isfinite(v).all() or (v.norm(dim=-1) < 1e-8).any() for v in values):
        raise ValueError('公开特征必须有限且非零')
    qi, target = F.normalize(query_features.detach(), dim=-1), F.normalize(target_feature.detach(), dim=-1)
    image = F.normalize(image_features.detach(), dim=-1)
    keys = F.normalize(image + F.normalize(text_features.detach(), dim=-1), dim=-1)
    scores, pure = qi @ keys.T, target.reshape(1, -1) @ image.T
    return dict(joint1=scores.max(-1).values, joint3=scores.topk(3, dim=-1).values[:, -1],
                image1=pure.max(-1).values, image3=pure.topk(3, dim=-1).values[:, -1])


def margin_loss(features, joint, target, thresholds, config=PublicMarginConfig()):
    scores = joint.scores(features)
    image = F.normalize(features.float(), dim=-1) @ F.normalize(target, dim=-1).T
    def loss(value, threshold):
        return config.temperature * F.softplus((threshold + config.margin - value) / config.temperature).mean()
    return (loss(scores, thresholds['joint3']) + config.top1_weight*loss(scores, thresholds['joint1'])
            + config.image_weight*(loss(image, thresholds['image3']) + config.top1_weight*loss(image, thresholds['image1'])))


def verify_public_manifest(manifest):
    if manifest.get('victim_gallery_read') is not False or not manifest.get('repository') or not manifest.get('revision'):
        raise ValueError('缺少独立公开数据来源声明')
    images, products = set(), set()
    for row in manifest['records']:
        path=Path(row['image'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['image_sha256']:
            raise ValueError('公开池图片改变')
        product = str(row['metadata']['id'])
        if row['pixel_sha256'] in images or product in products:
            raise ValueError('公开池图像或商品跨条目重复')
        images.add(row['pixel_sha256']); products.add(product)
    if not all(any(r['split']==s for r in manifest['records']) for s in ('train','development')):
        raise ValueError('公开训练和开发划分不全')


def exclude_own_duplicates(manifest, own_paths):
    own = set()
    for path in own_paths:
        with Image.open(relocate_io_path(path)) as opened:
            pixels=np.asarray(opened.convert('RGB'))
        own.add(hashlib.sha256(str(pixels.shape).encode()+pixels.tobytes()).hexdigest())
    if any(row['pixel_sha256'] in own for row in manifest['records']):
        raise ValueError('公开池包含自身源图或目标图，不能继续当前冻结抽样')


def select_public_hard_pool(records, image_features, text_features, queries, *, hard_count, random_count):
    """只用公开池和自身查询筛难例，背景覆盖按固定公开标识选取。"""
    if len(records) != len(image_features) or hard_count+random_count > len(records):
        raise ValueError('公开样本数与特征不一致')
    keys=F.normalize(F.normalize(image_features, dim=-1)+F.normalize(text_features, dim=-1), dim=-1)
    difficulty=(F.normalize(queries,dim=-1)@keys.T).max(0).values
    ordered=sorted(range(len(records)),key=lambda i:(-float(difficulty[i]), records[i]['id']))
    selected=ordered[:hard_count]
    remaining=sorted(ordered[hard_count:],key=lambda i:hashlib.sha256(records[i]['id'].encode()).hexdigest())
    return selected+remaining[:random_count]
