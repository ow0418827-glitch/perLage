"""三个平级方法的统一生成接口；完整实验先由公共筛选阶段选定图对。"""
import copy
import json
from src.paths import ProjectPath as Path

METHODS = ('lucid', 'PerLAGE', 'PerLAGE_no_pixel')
NATURAL_METHODS = ('PerLAGE', 'PerLAGE_no_pixel')


def normalize_method(method):
    if method == 'perlage':
        return 'PerLAGE'
    if method == 'perlage_no_pixel':
        return 'PerLAGE_no_pixel'
    if method not in METHODS:
        raise ValueError('方法必须是 lucid、PerLAGE 或 PerLAGE_no_pixel（无像素扰动感知方法）')
    return method


def add_method_arguments(parser):
    parser.add_argument('--method', choices=METHODS, default='PerLAGE')
    parser.add_argument('--perlage-config', help='感知标签保护生成编辑的公开训练配置')
    parser.add_argument('--perlage-no-pixel-config', help='无像素扰动感知方法的独立公开训练配置')


def options_from_args(args, group=None):
    if group is not None:
        raise ValueError('已移除按组分派配置和记忆的机制')
    method = normalize_method(args.method)
    config = getattr(args, 'perlage_config', None)
    no_pixel_config = getattr(args, 'perlage_no_pixel_config', None)
    if config and method != 'PerLAGE':
        raise ValueError('生成配置只能用于对应的感知方法')
    if no_pixel_config and method != 'PerLAGE_no_pixel':
        raise ValueError('无像素方法配置只能用于 PerLAGE_no_pixel')
    if config:
        return {'perlage_config': config}
    return {'perlage_no_pixel_config': no_pixel_config} if no_pixel_config else {}


def _adjacent_feature_text(target):
    """读取已冻结的特点正文；文件损坏或两份正文冲突时不能静默关闭监督。"""
    texts = []
    directory = Path(target).resolve().parent
    for name in ('definition.json', 'features.json'):
        path = directory / name
        if not path.is_file():
            continue
        document = json.loads(path.read_text(encoding='utf-8'))
        text = (document.get('target_feature_text') or document.get('alignment_canonical_text') or document.get('canonical_sentence')
                or document.get('feature_text') or document.get('target', {}).get('feature_text'))
        if text is not None:
            if not isinstance(text, str) or not text.strip():
                raise ValueError('冻结的目标商品特点文字必须为非空字符串')
            texts.append(text)
    if len(set(texts)) > 1:
        raise ValueError('图对定义与特点文件中的目标文字不一致')
    return texts[0] if texts else None


def generate_attack(source, target, output_dir, *, method='PerLAGE', steps=1000,
                    epsilon_levels=16, device='cuda', optimization_config=None,
                    perlage_config=None, perlage_no_pixel_config=None, target_text=None, text_weight=1.0):
    method = normalize_method(method)
    if method == 'lucid':
        if perlage_config or perlage_no_pixel_config:
            raise ValueError('像素方法不能读取感知方法配置')
        # 默认开启特点对齐；缺少正文时停止，不能退回无文字监督。
        frozen_text = _adjacent_feature_text(target)
        if target_text is None:
            target_text = frozen_text
        if target_text is None:
            raise ValueError('像素方法默认文字对齐需要选图阶段冻结的目标商品特点文字')
        if frozen_text is not None and target_text != frozen_text:
            raise ValueError('像素方法的文字与同一图对冻结的目标特点不一致')
        from src.visual._lucid.pipeline import generate_attack as generate_lucid
        image, metadata = generate_lucid(source, target, output_dir,
            optimization_config=optimization_config, steps=steps,
            epsilon_levels=epsilon_levels, device=device,
            target_text=target_text, text_weight=text_weight)
        from src.visual.image_utils import sha256_file
        metadata = dict(metadata, method='lucid', source_image_sha256=sha256_file(source),
                        target_image_sha256=sha256_file(target))
        if metadata.get('text_alignment'):
            metadata['text_alignment']['source'] = 'target_product_features'
    else:
        if method == 'PerLAGE':
            if perlage_no_pixel_config:
                raise ValueError('完整感知方法不能读取无像素方法配置')
            configured = perlage_config
            from src.visual._perlage.pipeline import generate_perlage as generator
        else:
            if perlage_config:
                raise ValueError('无像素方法必须使用其独立配置参数')
            configured = perlage_no_pixel_config
            from src.visual._perlage_no_pixel.pipeline import generate_perlage_no_pixel as generator
        if not configured:
            raise ValueError('感知方法缺少本次筛选图对的公开配置')
        definition = (copy.deepcopy(configured) if isinstance(configured, dict)
                      else json.loads(Path(configured).read_text(encoding='utf-8')))

        # 独立配置文件按其所在目录解析图片，不受启动时的工作目录影响。
        config_directory = None if isinstance(configured, dict) else Path(configured).resolve().parent
        for role, given in (('source', source), ('target', target)):
            value = definition.get('data', {}).get(role)
            if value and not Path(value).is_absolute():
                directory = config_directory or Path(given).resolve().parent
                definition['data'][role] = str((directory / value).resolve())

        # 统一文字对齐约束：感知方法必须通过公开配置绑定同一特点文字，严禁随意外传不匹配的文字指令
        configured_queries = definition.get('data', {}).get('training_queries') or []
        configured_feature_text = definition.get('target_feature_text') or (configured_queries[0] if configured_queries else None)
        if definition.get('target_feature_text') is not None and configured_queries != [configured_feature_text]:
            raise ValueError('感知方法的实际训练文字必须与冻结的目标商品特点完全一致')
        if target_text is not None:
            if not configured_feature_text or target_text != configured_feature_text:
                raise ValueError('感知方法只从其公开配置读取商品名和通用训练指令')
        if text_weight != 1.0:
            raise ValueError('感知方法只从其公开配置读取商品名和通用训练指令')

        from src.visual.image_utils import sha256_file
        image, ready = generator(source, target, output_dir, definition=definition, device=device)
        metadata = dict(method=method, definition=ready['definition'], status='success',
            output_image=str(image.resolve()), output_image_sha256=sha256_file(image),
            source_image_sha256=sha256_file(source), target_image_sha256=sha256_file(target),
            perlage_ready=ready, shadow_pool_used=False,
            completion_policy='four_natural_stages_with_perceptual_latent_optimization_complete')
    from src.visual.image_utils import write_json
    write_json(Path(output_dir) / 'generation_result.json', metadata)
    return image, metadata
