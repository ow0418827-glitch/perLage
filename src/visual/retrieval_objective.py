"""供保留方法共用的基础工具；不提供其他图片优化方法。"""
from __future__ import annotations

import hashlib
import math
from src.paths import ProjectPath as Path
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from src.visual.image_utils import sha256_file

def _query_digest(text):
    return hashlib.sha256(' '.join(text.split()).casefold().encode('utf-8')).hexdigest()


def purification_value(values, reduction='mean', temperature=0.1):
    """同一批真实净化目标的均值或归一化软最小值；后者优先改善弱路径。"""
    if reduction not in ('mean', 'softmin'):
        raise ValueError('未知净化目标归约')
    if isinstance(temperature, bool) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('净化软最小值温度必须有限且为正数')
    values = values.reshape(-1)
    if not values.numel() or not torch.isfinite(values).all():
        raise ValueError('净化目标不能为空或包含非有限数值')
    if reduction == 'mean':
        return values.mean()
    return -temperature * (torch.logsumexp(-values / temperature, dim=0)
                           - math.log(values.numel()))


TRAINING_DATA_POLICY = 'catalog_image_and_public_title_only_v1'


def objective_implementation_identity():
    """绑定实际目标函数及分块编码器；必需的代码摘要不得静默缺失。"""
    from src.visual.encoders import joint as joint_module
    paths = {'objective_source_sha256': Path(__file__),
             'encoder_source_sha256': Path(joint_module.__file__)}
    result = {}
    for key, path in paths.items():
        digest = sha256_file(path)
        if not isinstance(digest, str) or len(digest) != 64:
            raise FileNotFoundError('训练实现证据缺失：' + str(path))
        result[key] = digest
    return result


def validate_public_title(source_title):
    """训练只接收公开商品名，拒绝旧购买轨迹及其改名后的明显误传。"""
    if not isinstance(source_title, str) or not source_title.strip():
        raise ValueError('图片训练需要非空的公开商品名称 source_title')
    if any(c in source_title for c in ('\n', '\r')) or any(
            marker in source_title.casefold() for marker in
            ('instruction:', 'actions and observations:', 'observed purchase:', 'action: think[')):
        raise ValueError('source_title 只能是公开商品名称，不能是购买日志或操作轨迹')
    return source_title.strip()


def training_data(definition):
    """只放行明确的公开训练输入；旧历史配置须重建，不能悄悄复用。"""
    data = definition.get('data', {})
    allowed = {'source', 'target', 'source_title', 'training_queries'}
    if set(data) != allowed or definition.get('training_data_policy') != TRAINING_DATA_POLICY:
        raise ValueError('训练配置只能含图对、公开商品名和通用训练指令；禁止 source_history 或任何记忆库输入，请重建配置和公开池')
    return dict(data, source_title=validate_public_title(data['source_title']))


class JointRetrievalObjective:
    """使用公开商品名称和通用训练请求，按图片与名称各半构造代理键。

    查询图片固定为目标图片；训练请求应独立于正式测试请求。本类不读取
    记忆库排名或测试指标，增加的是联合余弦监督，不能保证最终排名居首。
    """
    def __init__(self, surrogate, target_feature, source_title, training_queries,
                 excluded_query_texts=None):
        from src.visual.encoders.joint import ChunkedClipEncoder

        source_title = validate_public_title(source_title)
        if not isinstance(training_queries, (list, tuple)) or not training_queries:
            raise ValueError('联合目标需要独立的非空训练请求列表')
        if any(not isinstance(text, str) or not text.strip() for text in training_queries):
            raise ValueError('训练请求必须是非空字符串')
        if excluded_query_texts is not None and (
                not isinstance(excluded_query_texts, (list, tuple))
                or any(not isinstance(text, str) for text in excluded_query_texts)):
            raise ValueError('排除请求必须是字符串列表')
        query_digests = [_query_digest(text) for text in training_queries]
        excluded_digests = {_query_digest(text) for text in excluded_query_texts or ()}
        if len(set(query_digests)) != len(query_digests):
            raise ValueError('训练请求不得重复')
        if set(query_digests) & excluded_digests:
            raise ValueError('训练请求与排除的正式测试请求重叠')
        encoder = ChunkedClipEncoder(SimpleNamespace(
            surrogate=surrogate, device=surrogate.device, snapshot=surrogate.snapshot()))
        title = encoder.encode_text(source_title)
        title_audit = dict(encoder.last_encoding_audit)
        queries, query_audits = [], []
        target_feature = F.normalize(target_feature.detach().float(), dim=-1)
        for text in training_queries:
            text_feature = torch.as_tensor(encoder.encode_text(text),
                dtype=torch.float32, device=surrogate.device).reshape(1, -1)
            # 正式分块检索器默认等权融合；保持相同的先归一化再相加顺序。
            queries.append(F.normalize(target_feature + text_feature, dim=-1))
            query_audits.append(dict(encoder.last_encoding_audit))
        self.title_feature = torch.as_tensor(title, dtype=torch.float32,
            device=surrogate.device).reshape(1, -1).detach()
        self.query_features = torch.cat(queries, dim=0).detach()
        self.audit = {'algorithm': 'unit_image_plus_public_product_title_v1',
            'training_data_policy': TRAINING_DATA_POLICY,
            'source_title_sha256': hashlib.sha256(source_title.encode('utf-8')).hexdigest(),
            'public_title_audit': title_audit, 'training_query_audits': query_audits,
            'training_query_normalized_sha256': query_digests,
            'excluded_query_normalized_sha256': sorted(excluded_digests),
            'query_image': 'target_image_unpurified', 'image_weight': 0.5,
            'victim_memory_used': False, 'purchase_history_used': False, 'test_ranking_used': False,
            'query_disjointness': 'normalized_exact_text_only_not_semantic_independence',
            **objective_implementation_identity()}

    def scores(self, image_features):
        joint = F.normalize(F.normalize(image_features.float(), dim=-1)
                            + self.title_feature, dim=-1)
        return joint @ self.query_features.T

    def value(self, image_features):
        return purification_value(self.scores(image_features), 'softmin', 0.04)
