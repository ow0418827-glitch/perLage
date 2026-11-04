"""两种独立方法的统一生成接口；完整实验先由公共筛选阶段选定图对。"""
import json
from pathlib import Path

METHODS = ('lucid', 'PerLAGE')
NATURAL_METHODS = ('PerLAGE',)


def normalize_method(method):
    if method == 'perlage':
        return 'PerLAGE'
    if method not in METHODS:
        raise ValueError('方法必须是 lucid（原始像素优化）或 PerLAGE（感知标签保护生成编辑）')
    return method


def add_method_arguments(parser):
    parser.add_argument('--method', choices=METHODS, default='PerLAGE')
    parser.add_argument('--perlage-config', help='感知标签保护生成编辑的公开训练配置')


def options_from_args(args, group=None):
    if group is not None:
        raise ValueError('已移除按组分派配置和记忆的机制')
    method = normalize_method(args.method)
    config = getattr(args, 'perlage_config', None)
    if config and method != 'PerLAGE':
        raise ValueError('生成配置只能用于对应的感知方法')
    return {'perlage_config': config} if config else {}


def generate_attack(source, target, output_dir, *, method='PerLAGE', steps=1000,
                    epsilon_levels=16, device='cuda', optimization_config=None,
                    perlage_config=None, target_text=None, text_weight=1.0):
    method = normalize_method(method)
    if method == 'lucid':
        if perlage_config:
            raise ValueError('像素方法不能读取感知方法配置')
        from src.vision.dual_rag_bridge import generate_attack as generate_lucid
        image, metadata = generate_lucid(source, target, output_dir,
            optimization_config=optimization_config, steps=steps,
            epsilon_levels=epsilon_levels, device=device,
            target_text=target_text, text_weight=text_weight)
        from src.vision.image_utils import sha256_file
        metadata = dict(metadata, method='lucid', source_image_sha256=sha256_file(source),
                        target_image_sha256=sha256_file(target))
    else:
        if not perlage_config:
            raise ValueError('感知方法缺少本次筛选图对的公开配置')
        if target_text is not None or text_weight != 1.0:
            raise ValueError('感知方法只从其公开配置读取商品名和通用训练指令')
        from src.vision.perlage import generate_perlage
        from src.vision.image_utils import sha256_file
        definition = (dict(perlage_config) if isinstance(perlage_config, dict)
                      else json.loads(Path(perlage_config).read_text(encoding='utf-8')))
        image, ready = generate_perlage(source, target, output_dir, definition=definition, device=device)
        metadata = dict(method='PerLAGE', definition=ready['definition'], status='success',
            output_image=str(image.resolve()), output_image_sha256=sha256_file(image),
            source_image_sha256=sha256_file(source), target_image_sha256=sha256_file(target),
            perlage_ready=ready, shadow_pool_used=False,
            completion_policy='four_natural_stages_with_perceptual_latent_optimization_complete')
    from src.vision.image_utils import write_json
    write_json(Path(output_dir) / 'generation_result.json', metadata)
    return image, metadata
