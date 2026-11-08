"""独立无像素扰动感知优化对抗方法 (PerLAGE_no_pixel) 的完成证据与审计。"""
import json
from src.paths import ProjectPath as Path

from src.visual._perlage.components.evidence import STAGES
from src.visual.image_utils import sha256_file
from src.visual.retrieval_objective import TRAINING_DATA_POLICY

METHOD = 'PerLAGE_no_pixel'
DEFINITION = 'perlage_no_pixel_foreground_statistics_harmonization_perceptual_latent_v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def definition_for_method(method=METHOD):
    if method in (METHOD, 'perlage_no_pixel'):
        return DEFINITION
    raise ValueError('未知的无像素感知方法身份：' + str(method))


def validate_complete(ready, image=None, *, method=None):
    if not isinstance(ready, dict):
        raise ValueError('完成记录必须为字典对象')
    if method is None:
        method = ready.get('method', METHOD)
    if method not in (METHOD, 'perlage_no_pixel'):
        raise ValueError(f'方法身份不符合无像素感知方法，期望 {METHOD}，实际 {method}')
    if ready.get('method') != METHOD:
        raise ValueError(f'完成记录中的方法身份不符合 {METHOD}，实际 {ready.get("method")}')
    if ready.get('definition') != DEFINITION:
        raise ValueError(f'完成记录中的方法定义不符合 {DEFINITION}，实际 {ready.get("definition")}')
    if ready.get('status') != 'completed':
        raise ValueError('无像素感知方法尚未标记为已完成')
    if ready.get('color_scheme') != 'statistics':
        raise ValueError('无像素感知方法仍保留主体统计选色')

    # Strict zero-pixel and zero-residual constraints
    if ready.get('pixel_optimization') is not False:
        raise ValueError('无像素方法不能包含像素优化')
    if ready.get('auxiliary_max_levels', 0) != 0:
        raise ValueError('无像素方法不能包含固定像素残差')
    if ready.get('latent_config', {}).get('auxiliary_levels', 0) != 0:
        raise ValueError('无像素方法潜变量配置不能包含辅助像素预算')
    if ready.get('optimization_mode') not in (None, 'latent_only'):
        raise ValueError('无像素方法优化模式必须为 latent_only')

    opt_identity_path = ready.get('initialization_optimizer_identity')
    if opt_identity_path and Path(opt_identity_path).is_file():
        opt = read(opt_identity_path)
        opt_config = opt.get('config', {})
        if any(opt_config.get(k, 0) != 0 for k in ('pixel_clean_steps', 'pixel_diffusion_steps', 'pixel_levels')):
            raise ValueError('无像素方法不能包含像素优化步数或等级')

    method_identity = ready.get('method_identity')
    if (method_identity and Path(method_identity).is_file()
            and ready.get('method_identity_sha256') == sha256_file(method_identity)):
        binding = read(method_identity)
        if (binding.get('method') != METHOD or binding.get('definition') != DEFINITION
                or binding.get('configuration_sha256') != ready.get('configuration_sha256')
                or any(binding.get(name + '_sha256') != ready.get(name + '_sha256')
                       for name in ('source', 'target'))):
            raise ValueError('完成记录未绑定本次独立方法及公开输入')

    for name in ('shadow_pool_used', 'purchase_history_used', 'victim_memory_used', 'formal_queries_used'):
        if ready.get(name) is not False:
            raise ValueError('无像素感知方法缺少独立训练边界：' + name)
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
