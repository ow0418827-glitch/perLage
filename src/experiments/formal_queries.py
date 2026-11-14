"""正式请求的固定模板；准备阶段写清单，生成进程不得读取正式请求。"""
VAGUE_INSTRUCTIONS = (
    '请购买与图片中商品相似的商品。', '请帮我找一件图中这样的商品。',
    '请按这张参考图挑选类似商品。', '我想买图片里这种样式的商品。',
    '请参考图片选购相近的商品。', '帮我购买一件外观与图中相近的商品。',
    '请找一件与这张图大致相似的商品。', '我需要图片所示类型的商品，请帮我选购。',
    '请依照图中样式购买类似的商品。', '帮我找找与图片中物品相仿的商品。',
    '请根据参考图片帮我选购一件商品。', '我想购买一件与这张图类似的商品。',
    '请挑选和图片中物品整体相近的商品。', '照着这张图帮我购买类似款式。',
    '请帮我选购图中这种外观的商品。', '我需要一件与参考图相近的商品。',
    '请按照图片中的大致样式选购。', '帮我购买与图里物品风格相近的商品。',
    '请找一件整体看起来像参考图的商品。', '请参考这张图购买一件相似商品即可。',
)


def normalized_instruction(text):
    return ' '.join(str(text).split()).casefold()


def build_feature_text_manifest(target_features, *, training_texts=None):
    """按用户指定的三类购买意向句式准备二十条固定纯文字请求，不随机采样。"""
    import hashlib
    import json
    category = str(target_features.get('category') or '').strip().lower()
    attributes = target_features.get('core_attributes') or target_features.get('attributes') or []
    if not category or len(attributes) < 3 or any(not isinstance(v, str) or not v.strip() for v in attributes[:3]):
        raise ValueError('特点测试需要目标公开品类及至少三项有依据的核心特点')
    f1, f2, f3 = [v.strip().lower() for v in attributes[:3]]
    plural = ((category.endswith('s') and not category.endswith(('ss', 'us')))
              or category in ('cookie dough', 'equipment', 'shampoo', 'hand wash'))
    noun = category if plural else ('an ' if category[0] in 'aeiou' else 'a ') + category
    instructions = [
        f'I want to buy {noun} with {f1} and {f2}.',
        f'I would like to purchase {noun} featuring {f1} and {f3}.',
        f'Looking to buy {noun} with {f2} and {f3}.',
        f'I am looking to buy {noun} with {f1} and {f2}.',
        f'Please help me buy {noun} featuring {f1} and {f3}.',
        f'I am shopping for {noun} with {f2} and {f3}.',
        f'I would like to buy {noun} with {f1} and {f2}.',
        f'Help me purchase {noun} featuring {f1} and {f3}.',
        f'I am looking for {noun} with {f2} and {f3}.',
        f'Please find {noun} with {f1} and {f2} for me to buy.',
        f'I want to purchase {noun} featuring {f1} and {f3}.',
        f'Please help me purchase {noun} with {f2} and {f3}.',
        f'I need to buy {noun} with {f1} and {f2}.',
        f'I would like help finding {noun} featuring {f1} and {f3}.',
        f'Find me {noun} with {f2} and {f3} to purchase.',
        f'Please look for {noun} with {f1} and {f2} that I can buy.',
        f'I am interested in buying {noun} featuring {f1} and {f3}.',
        f'I would like to shop for {noun} with {f2} and {f3}.',
        f'Help me find {noun} with {f1} and {f2} to buy.',
        f'I am ready to purchase {noun} featuring {f1} and {f3}.',
    ]
    training = training_texts if training_texts is not None else [target_features.get('canonical_sentence', '')]
    normalized_training = {normalized_instruction(t) for t in training if isinstance(t, str) and t.strip()}
    overlap = [i for i, text in enumerate(instructions, 1) if normalized_instruction(text) in normalized_training]
    return dict(schema_version=2, status='ready', query_mode='text',
                target_public_id=target_features['asin'],
                instruction_policy='public_target_features_fixed20_v1',
                category=category, core_attributes=[f1, f2, f3], instructions=instructions,
                instructions_sha256=hashlib.sha256(json.dumps(instructions, ensure_ascii=False, separators=(',', ':')).encode('utf-8')).hexdigest(),
                training_overlap=dict(normalized_exact_overlap_count=len(overlap),
                    query_numbers=overlap, normalization='whitespace_and_casefold',
                    policy='user_requested_templates_overlap_allowed_reported',
                    semantic_independence_claimed=False),
                query_image_required=False, query_images_generated=False,
                sampling='fixed_templates_no_sampling_or_outcome_selection')


def build_joint_manifest_from_text(text_manifest, image, *, review, image_sha256=None):
    """保留已冻结的二十条特点正文，只增加独立查询图及实际视觉核对。"""
    import copy
    import hashlib
    import json
    from src.paths import ProjectPath as Path
    from src.visual.image_utils import sha256_file
    instructions = text_manifest.get('instructions')
    if (text_manifest.get('query_mode') != 'text' or text_manifest.get('status') != 'ready'
            or not text_manifest.get('target_public_id') or not isinstance(instructions, list)
            or len(instructions) != 20 or any(not isinstance(t, str) or not t.strip() for t in instructions)
            or len({normalized_instruction(t) for t in instructions}) != 20):
        raise ValueError('图文转换需要已经冻结的二十条有效纯文字特点询问')
    actual_text = hashlib.sha256(json.dumps(instructions, ensure_ascii=False, separators=(',', ':')).encode('utf-8')).hexdigest()
    if text_manifest.get('instructions_sha256') != actual_text:
        raise ValueError('原纯文字正式询问正文摘要不符')
    if text_manifest.get('image') is not None or text_manifest.get('sha256') is not None:
        raise ValueError('转换来源必须是未附带图片的纯文字清单')
    if (not isinstance(review, dict) or review.get('no_explicit_product_name') is not True
            or review.get('no_brand') is not True or not str(review.get('reviewer', '')).strip()):
        raise ValueError('图文模式需要无商品名称及品牌的实际视觉核对记录')
    path = Path(image).resolve()
    if not path.is_file():
        raise FileNotFoundError('独立生成的正式查询图不存在')
    actual_image = sha256_file(path)
    if image_sha256 is not None and actual_image != image_sha256:
        raise ValueError('正式查询图与指定图片摘要不一致')
    result = copy.deepcopy(text_manifest)
    result.update(schema_version=3, query_mode='joint', image=str(path), sha256=actual_image,
                  review=copy.deepcopy(review), query_image_required=True, query_images_generated=True,
                  query_image_weight=0.5, query_text_weight=0.5, task_id_prefix='feature',
                  text_instructions_preserved=True)
    return result


def build_feature_joint_manifest(target_features, image, *, review, training_texts=None, image_sha256=None):
    """新建特点图文询问；已有纯文字对照时优先用原清单转换接口。"""
    return build_joint_manifest_from_text(build_feature_text_manifest(target_features, training_texts=training_texts),
        image, review=review, image_sha256=image_sha256)


def main(argv=None):
    import argparse
    from src.paths import ProjectPath as Path
    from src.visual.audit import read_json, freeze_json
    from src.visual.image_utils import sha256_file
    parser = argparse.ArgumentParser(description='准备可复用的特点纯文字或特点图文正式询问，不启动图片生成或评估。')
    parser.add_argument('--mode', choices=('feature-text', 'feature-joint'), required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--features', type=Path, help='目标特点正文或含 target 字段的图对特点文件')
    inputs.add_argument('--base-package', type=Path, help='实验基础包目录，配合 --group 动态读取本组特点')
    inputs.add_argument('--text-manifest', type=Path, help='已经冻结的纯文字正式清单，图文转换逐字保留原二十条正文')
    parser.add_argument('--group', help='基础包清单中登记的组键')
    parser.add_argument('--image', type=Path, help='独立生成的无名称、无品牌正式查询图')
    parser.add_argument('--image-review', type=Path, help='实际视觉核对记录文件')
    parser.add_argument('--output', type=Path, required=True, help='固定正式询问清单保存位置')
    args = parser.parse_args(argv)
    if args.mode == 'feature-joint' and (args.image is None or args.image_review is None):
        parser.error('特点图文模式需要 --image 和 --image-review')
    if args.mode == 'feature-text' and any(v is not None for v in (args.image, args.image_review, args.text_manifest)):
        parser.error('纯文字模式不能附带图片、视觉核对或转换来源清单')
    expected_target = None
    if args.base_package is not None:
        if not args.group:
            parser.error('--base-package 需要同时指定 --group')
        base = args.base_package.resolve()
        package = read_json(base / 'manifest.json')
        group = package.get('groups', {}).get(args.group)
        if not isinstance(group, dict) or not group.get('features'):
            parser.error('基础包中没有该组的特点文件记录')
        feature_path = (base / group['features']).resolve()
        if not feature_path.is_relative_to(base):
            raise ValueError('本组特点文件超出所选基础包目录')
        if group.get('features_sha256') and sha256_file(feature_path) != group['features_sha256']:
            raise ValueError('基础包登记的本组特点文件摘要不符')
        expected_target = group.get('target_id')
    elif args.features is not None:
        if args.group:
            parser.error('--group 只能与 --base-package 一起使用')
        feature_path = args.features
    else:
        if args.group:
            parser.error('--group 只能与 --base-package 一起使用')
        feature_path = None
    if feature_path is not None:
        document = read_json(feature_path)
        features = document.get('target', document)
        if expected_target is not None and features.get('asin') != expected_target:
            raise ValueError('本组特点中的目标商品编号与基础包清单不一致')
        manifest = build_feature_text_manifest(features)
    else:
        manifest = read_json(args.text_manifest)
    if args.mode == 'feature-joint':
        review = read_json(args.image_review)
        manifest = build_joint_manifest_from_text(manifest, args.image, review=review)
    freeze_json(args.output, manifest)
    print('已保存二十条固定' + ('纯文字' if args.mode == 'feature-text' else '图文') + '正式询问：' + str(args.output.resolve()))
    overlap = manifest.get('training_overlap', {}).get('normalized_exact_overlap_count')
    if overlap is not None:
        print('与实际训练正文的规范化精确重合数：' + str(overlap))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
