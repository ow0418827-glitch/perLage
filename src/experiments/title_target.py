"""完整原标题身份：lucid 继续使用 legacy_pixel 公开标识，支持完整标题监督。"""
import hashlib
import json
import math
from src.paths import ProjectPath as Path


def add_alignment_arguments(parser):
    parser.add_argument('--target-text-mode', choices=['none', 'title'], default='none',
        help='lucid (legacy_pixel)：none 保留纯图片目标，title 使用乙商品完整原标题')
    parser.add_argument('--text-weight', type=float, default=1.0,
        help='标题对齐的正权重；仅在 --target-text-mode title 时使用，默认 1.0')


def alignment_options(args):
    mode = getattr(args, 'target_text_mode', 'none')
    weight = getattr(args, 'text_weight', 1.0)
    if mode not in ('none', 'title'):
        raise ValueError('Invalid target text mode')
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
        raise ValueError('text_weight must be positive and finite')
    if mode == 'none':
        if weight != 1.0:
            raise ValueError('text_weight requires --target-text-mode title')
        return {}
    method = getattr(args, 'method', 'legacy_pixel')
    if method not in ('legacy_pixel', 'lucid'):
        raise ValueError('Title alignment is supported only by legacy_pixel (lucid)')
    # Include both provenance construction and the local encoding adapter in identity.
    paths = [Path(__file__), Path(__file__).parents[1] / 'vision/dual_rag_bridge.py',
             Path(__file__).parents[1] / 'vision/title_alignment.py']
    if method == 'papo':
        paths.append(Path(__file__).parents[1] / 'vision/papo_attack.py')
    implementation = hashlib.sha256()
    for path in paths:
        implementation.update(path.name.encode('utf-8'))
        implementation.update(path.read_bytes())
    options = {'target_text_mode': 'title', 'text_weight': float(weight),
               'title_target_implementation_sha256': implementation.hexdigest()}
    # Historical lucid identities did not carry a method key. Keep their shape
    # and interpret missing attack_method as legacy_pixel when reading them.
    if method == 'papo':
        options['attack_method'] = method
    return options


def alignment_attack_kwargs(identity):
    """从已绑定实际乙商品的身份透传原文；纯图片模式不增加调用参数。"""
    if not identity:
        return {}
    return {'target_text': identity['target_text'], 'text_weight': identity['text_weight']}


def title_identity(args, product, asin, source):
    options = alignment_options(args)
    if not options:
        return {}
    if product.get('asin') != asin:
        raise ValueError('Title product ID differs from the actual optimization target B')
    field = next((key for key in ('Title', 'name')
                  if isinstance(product.get(key), str) and product[key].strip()), None)
    if field is None:
        raise ValueError('Target B has no nonempty product title')
    title = product[field]  # Preserve original spelling, punctuation, units and whitespace.
    return {**options, 'target_asin': asin, 'target_text': title,
            'target_text_sha256': hashlib.sha256(title.encode('utf-8')).hexdigest(),
            'target_text_source': {**source, 'field': field}}


def catalog_title_identity(args, asin):
    if not alignment_options(args):
        return {}
    if not isinstance(asin, str) or not asin:
        raise ValueError('Title alignment requires the actual target B product ID')
    from src.environment.disk_catalog import DiskCatalog
    with DiskCatalog.open(args.catalog) as catalog:
        return title_identity(args, catalog.product(asin), asin,
            {'kind': 'disk_catalog', 'bundle_id': catalog.data_manifest['bundle_id']})


def pairing_title_identity(args, pairing):
    if not alignment_options(args):
        return {}
    path = Path(args.pairing_config).resolve()
    # Pairing validation canonicalizes names for the memory protocol. Read the
    # bound product's original record here so title supervision preserves every
    # original character without changing that established protocol.
    original = json.loads(path.read_text(encoding='utf-8'))
    product = original.get('requested_product', {})
    actual_asin = pairing['requested_product']['asin']
    raw_asin = product.get('asin')
    if not isinstance(raw_asin, str) or raw_asin.strip() != actual_asin:
        raise ValueError('Title product ID differs from the actual optimization target B')
    return title_identity(args, {**product, 'asin': actual_asin}, actual_asin,
        {'kind': 'pairing.requested_product', 'path': str(path)})


def validate_alignment_metadata(metadata, identity):
    """A completed image-only result must never impersonate a title-guided one."""
    if not identity:
        return
    title = identity.get('target_text')
    if (not isinstance(title, str) or not title.strip()
            or identity.get('target_text_sha256') != hashlib.sha256(title.encode('utf-8')).hexdigest()):
        raise ValueError('Frozen title alignment text digest does not match its title')
    expected_method = identity.get('attack_method', 'legacy_pixel')
    if (expected_method not in ('legacy_pixel', 'lucid')
            or metadata.get('method') != expected_method
            or metadata.get('target_text') != identity['target_text']
            or metadata.get('attack', {}).get('text_weight') != identity['text_weight']):
        raise ValueError('Attack title or text weight differs from frozen title alignment')
    alignment = metadata.get('text_alignment', {})
    if (not isinstance(alignment, dict) or alignment.get('enabled') is not True
            or alignment.get('text_sha256') != identity['target_text_sha256']
            or alignment.get('weight') != identity['text_weight']):
        raise ValueError('Attack is missing matching title alignment provenance')


def alignment_cli_options(args):
    options = alignment_options(args)
    return (['--target-text-mode', 'title', '--text-weight', str(options['text_weight'])]
            if options else [])
