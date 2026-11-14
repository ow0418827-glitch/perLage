"""独立自然优先感知优化对抗方法 (PerLAGE) 的完成证据与审计。"""
import json
from pathlib import Path

from src.vision._natural.evidence import STAGES
from src.vision.image_utils import sha256_file
from src.vision.retrieval_objective import TRAINING_DATA_POLICY

METHOD = 'PerLAGE'
DEFINITION = 'perlage_foreground_statistics_harmonization_perceptual_latent_v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_complete(ready, image=None):
    if (ready.get('method') != METHOD or ready.get('definition') != DEFINITION
            or ready.get('status') != 'completed' or ready.get('color_scheme') != 'statistics'):
        raise ValueError('不是完成的自然优先感知优化对抗方法')
    for name in ('shadow_pool_used', 'purchase_history_used', 'victim_memory_used', 'formal_queries_used'):
        if ready.get(name) is not False:
            raise ValueError('自然优先优化方法缺少独立训练边界：' + name)
    if ready.get('training_data_policy') != TRAINING_DATA_POLICY:
        raise ValueError('优化方法只接受公开图对与商品名训练')
    stages = ready.get('stages', {})
    if set(stages) != set(STAGES):
        raise ValueError('优化方法仍须完成四个自然优先阶段')
    for name, record in stages.items():
        if record.get('status') != 'completed' or not record.get('files'):
            raise ValueError('优化方法阶段没有完成：' + name)
        for path, digest in record['files'].items():
            if not Path(path).is_file() or sha256_file(path) != digest:
                raise ValueError('优化方法阶段证据缺失或改变：' + name)
    latent = stages['latent_generation_refinement']
    if (type(latent.get('steps_completed')) is not int or latent['steps_completed'] < 1
            or type(latent.get('steps_requested')) is not int
            or latent['steps_completed'] != latent['steps_requested']):
        raise ValueError('优化方法生成潜变量细化日程未完成')
    if (not isinstance(ready.get('initialization_evidence'), dict)
            or len(ready['initialization_evidence']) < 3
            or not ready.get('initialization_identity')
            or not ready.get('initialization_optimizer_identity')):
        raise ValueError('缺少本次独立初始化的完整身份及证据')
    for path, digest in ready['initialization_evidence'].items():
        if sha256_file(path) != digest:
            raise ValueError('优化方法初始化证据改变')
    if 'initialization_identity' in ready:
        identity = read(ready['initialization_identity'])
        if identity.get('public_binding', {}).get('shadow_pool_used') is not False:
            raise ValueError('优化方法不能复用含竞争池的初始化')
    final = ready.get('candidates', {}).get('final', {})
    if (not final or not Path(final.get('image', '')).is_file()
            or not isinstance(final.get('sha256'), str)
            or sha256_file(final['image']) != final['sha256']):
        raise ValueError('优化方法最终图片不符')
    if image is not None and sha256_file(image) != final['sha256']:
        raise ValueError('当前图片没有绑定优化方法完整完成记录')
    return ready
