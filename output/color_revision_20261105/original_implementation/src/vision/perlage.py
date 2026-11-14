"""独立自然优先感知优化对抗方法 (PerLAGE)；保持四阶段，释放感知优化预算。"""
import hashlib
from pathlib import Path

from src.vision._perlage.pipeline import (
    PerLAGEConfig, generate_candidate, validate_definition)
from src.vision._perlage.evidence import validate_complete
from src.vision.image_utils import sha256_file
from src.vision.retrieval_objective import training_data

__all__ = ['generate_perlage', 'PerLAGEConfig', 'validate_definition']


def generate_perlage(source, target, output, *, definition, device='cuda'):
    """从公开图对、名称和几何提示生成完整优化对抗图片，接口不读取实验或私有评估。"""
    validate_definition(definition)
    data = training_data(definition)
    image, ready = generate_candidate(source, target, output,
                                          definition=definition, device=device)
    validate_complete(ready, image)
    for name, path in (('source', source), ('target', target)):
        if ready[name + '_sha256'] != sha256_file(path):
            raise ValueError('已有优化对抗成图的公开输入改变：' + name)
    title_sha = hashlib.sha256(data['source_title'].encode('utf-8')).hexdigest()
    if ready['source_title_sha256'] != title_sha:
        raise ValueError('已有优化对抗成图不属于当前公开商品名')
    return Path(image), ready
