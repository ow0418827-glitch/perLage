"""完整入口：公共图对筛选 → 按方法生成 → 冻结 → 共享池评估 → 复算报告。"""
import argparse
import copy
from dataclasses import asdict
import os
from src.paths import PUBLIC, PRIVATE, VISUAL_CONFIGS, EXPERIMENT_CONFIGS, public_root_for, ProjectPath as Path
import shutil

from src.paths import PROJECT, CONFIGS, DATA, OUTPUT, PRIVATE, SHARED_MEMORY
from src.visual.audit import digest, freeze_json, read_json, save_json
from src.visual.image_utils import sha256_file
from src.visual.optimization import METHODS, NATURAL_METHODS, normalize_method
from src.visual.catalog_training_guard import catalog_training_guard
from src.visual.pair_selection import SelectionConfig, LocalSelectionBackend, choose_pair, validate_product
from src.experiments.query_contract import QueryImagePending, pending_query_spec


class SelectedAnnotationsPending(ValueError):
    """先完成轻量抽样，再为实际选中的图对准备公开生成标注。"""


TRAINING_QUERIES = (
    'Find a product with a similar overall visual appearance.',
    'Use the reference picture as a broad visual guide for product matching.',
    'Look for an item with a comparable general shape and finish.',
)


def project_path(value):
    p = Path(value)
    return (p if p.is_absolute() else PROJECT / p).resolve()


def source_snapshot():
    # 只记录公共生成及选择所依赖的实现，不在训练进程读取评估请求源码。
    files = [PROJECT / 'src/paths.py', PROJECT / 'src/model_identity.py', Path(__file__)]
    for directory in ('src/visual',):
        files.extend(p for p in (PROJECT / directory).rglob('*.py') if '__pycache__' not in p.parts)
    sources = {p.relative_to(PROJECT).as_posix(): sha256_file(p) for p in sorted(files)}
    sources['src/visual/config/optimization.json'] = sha256_file(VISUAL_CONFIGS / 'optimization.json')
    return sources


def load_config(path):
    config = read_json(path)
    allowed = {'schema_version', 'method', 'public_catalog', 'public_pair', 'selection', 'selection_models',
               'generation', 'evaluation', 'run_dir'}
    if not isinstance(config, dict) or set(config) - allowed or config.get('schema_version') != 1:
        raise ValueError('完整流程配置含未知字段或版本无效')
    normalize_method(config.get('method', 'PerLAGE'))
    if bool(config.get('public_catalog')) == bool(config.get('public_pair')):
        raise ValueError('须指定公开候选清单或已确定的分组图对，且只能使用其中一种输入')
    if any(k in config.get('evaluation', {}) for k in ('groups', 'group_memory', 'memory_by_group')):
        raise ValueError('完整流程不再接受按组记忆配置')
    if config.get('evaluation', {}).get('query_mode', 'joint') not in ('text', 'joint'):
        raise ValueError('正式查询模式只能是纯文字或图文联合')
    if config.get('evaluation', {}).get('defense_mode', 'panel') not in ('none', 'panel'):
        raise ValueError('防御范围只能为无防御或既有完整面板')
    return config


def public_inputs(config):
    path = project_path(config['public_catalog'])
    public_root = public_root_for(path, extra_roots=(PUBLIC, DATA))
    document = read_json(path)
    if set(document) != {'schema_version', 'target', 'candidates'} or document['schema_version'] not in (1,2):
        raise ValueError('公开清单仅接受目标商品及攻击者自有候选')
    target = validate_product(document['target'], path.parent,generation_ready=False,public_root=public_root)
    candidates = [validate_product(r, path.parent, owned=True,generation_ready=False,public_root=public_root) for r in document['candidates']]
    for record in [target] + candidates:
        for key in ('image', 'foreground_mask'):
            if record.get(key) and not Path(record[key]).is_relative_to(public_root):
                raise ValueError('公开图片和标注须置于新项目公开目录；禁止关联旧实验或私有图库')
    return path, target, candidates


def selected_feature_text(selection):
    """统一解析图对绑定的特点正文，拒绝两处保存了不同正文。"""
    target = selection['target']
    texts = [text for text in (target.get('feature_text'), target.get('canonical_sentence'),
             selection.get('target_feature_text'), selection.get('feature_text')) if text is not None]
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError('目标商品特点文字必须为非空字符串')
    if len(set(texts)) > 1:
        raise ValueError('图对中保存的目标商品特点文字不一致')
    return texts[0] if texts else None


def bind_selected_features(selection):
    """仅在图对确定后提取特点；旧发布图对只在新运行副本中补充，不改历史清单。"""
    if selected_feature_text(selection) is None:
        from src.visual.target_features import extract_product_features
        try:
            features = extract_product_features(selection['target']['id'])
            selection['target_feature_text'] = features['canonical_sentence']
            selection['target'].update(brand=features['brand'], attributes=features['attributes'],
                                       feature_text=features['canonical_sentence'],
                                       training_queries=features['training_queries'])
        except (KeyError, ValueError, FileNotFoundError):
            pass
    return selection


def select_pair(config, root, *, device='cuda', backend_factory=LocalSelectionBackend):
    evaluation = config.get('evaluation', {})
    protected = [project_path(evaluation.get('shared_pool', SHARED_MEMORY)),
                 project_path(evaluation.get('query_manifest', PRIVATE / 'query_image.json')),
                 project_path(evaluation.get('encoder_config', PRIVATE / 'evaluator.json'))]
    root = Path(root)
    public_input = project_path(config.get('public_pair') or config['public_catalog'])
    input_root = public_root_for(public_input, extra_roots=(PUBLIC, DATA))
    allowed_roots = [root, input_root, PUBLIC]
    cal_root = None
    raw_settings = dict(config.get('selection', {}))
    if not config.get('public_pair') and raw_settings.get('calibration_path'):
        calibration = project_path(raw_settings['calibration_path'])
        cal_root = public_root_for(calibration, extra_roots=(PUBLIC, DATA))
        if cal_root not in allowed_roots:
            allowed_roots.append(cal_root)
    with catalog_training_guard(allowed_output_roots=allowed_roots, private_paths=protected):
        if config.get('public_pair'):
            from src.visual.pair_assets import load_pair
            manifest = project_path(config['public_pair'])
            # 图对已经冻结时核验图像及公开筛选证据，无需再次加载选图模型权重。
            selected = load_pair(manifest, input_root, verify_models=False)
            if config.get('selection') and digest(selected['configuration']) != digest(config['selection']):
                raise ValueError('分组图对已确定；更改筛选参数须重新执行公开选图')
            original_models = selected.get('binding', {}).get('models_configuration', {})
            if config.get('selection_models', original_models) != original_models:
                raise ValueError('分组图对已确定；更改公开代理须重新执行公开选图')
            selected['scoring_binding'] = selected['binding']
            selected['binding'] = dict(public_pair_sha256=selected['pair_manifest_sha256'],
                                       scoring_record_sha256=selected['scoring_record_sha256'])
            bind_selected_features(selected)
            freeze_json(root / 'selection' / 'SELECTED_PAIR.json', selected)
            return read_json(root / 'selection' / 'SELECTED_PAIR.json')
        raw_settings=dict(config.get('selection',{}))
        if raw_settings.get('calibration_path'):
            calibration=project_path(raw_settings['calibration_path'])
            public_root_for(calibration, extra_roots=(PUBLIC, DATA))
            raw_settings['calibration_path']=str(calibration)
        settings = SelectionConfig(**raw_settings)
        settings.validate()
        path, target, candidates = public_inputs(config)
        binding = dict(public_catalog_sha256=sha256_file(path), public_records_sha256=digest([target] + candidates),
                       configuration=digest(asdict(settings)), models_configuration=config.get('selection_models', {}),
                       selector_code_sha256=sha256_file(PROJECT / 'src/visual/pair_selection.py'),
                       calibration_sha256=sha256_file(settings.calibration_path))
        saved_path = root / 'selection' / 'SELECTED_PAIR.json'
        if saved_path.exists():
            saved = read_json(saved_path)
            if saved['binding'] != binding:
                raise ValueError('公开候选、筛选配置或实现改变；请使用新的运行目录')
            if saved.get('status') != 'selected':
                raise ValueError('该公开候选集没有合格图对')
            from src.visual.audit import verify_model_snapshot
            verify_model_snapshot(saved['models'])
            return saved
        cal_roots = tuple(dict.fromkeys([r for r in (cal_root, input_root, PUBLIC, DATA) if r is not None]))
        backend = backend_factory(config.get('selection_models', {}), root / 'selection' / 'features', device)
        try:
            selection = choose_pair(target, candidates, backend, settings,
                                    public_root=input_root, calibration_public_root=cal_roots)
        finally:
            backend.close()
        selection.update(binding=binding, target=target)
        if selection['selected_source_id']:
            selection['source'] = next(c for c in candidates if c['id'] == selection['selected_source_id'])
            pending_path=root/'selection'/'DRAWN_PAIR.json'
            if pending_path.exists():
                pending=read_json(pending_path)
                if any(pending.get(k)!=selection.get(k) for k in ('configuration','models','candidates','draw','selected_source_id')):
                    raise ValueError('已抽中的图对只能补充标注；候选、模型或抽样规则改变须使用新的运行目录')
                for role in ('source','target'):
                    if any(pending[role].get(k)!=selection[role].get(k) for k in ('id','title','image_sha256','type_review')):
                        raise ValueError('补充标注不能更换已抽中的公开商品身份或图片')
                selection['annotation_binding']=dict(drawn_pair_sha256=sha256_file(pending_path))
            if any(p.get('text_regions_reviewed') is not True or not (p.get('boxes') or p.get('foreground_mask'))
                   for p in (selection['source'],target)):
                if not pending_path.exists():freeze_json(pending_path,selection)
                raise SelectedAnnotationsPending('轻量选图已完成；仅须核对选中源图与目标图的主体和印刷区，然后继续发布或生成')
            selection['source']=validate_product(selection['source'],path.parent,owned=True,public_root=input_root)
            selection['target']=validate_product(target,path.parent,public_root=input_root)
            bind_selected_features(selection)
            from src.visual.pair_assets import publish_pair
            scoring_path = root / 'selection' / 'SCORED_PAIR.json'
            freeze_json(scoring_path, selection)
            selection = publish_pair(scoring_path, root / 'data' / 'pair', root / 'data', input_public_root=input_root)
        freeze_json(saved_path, selection)
        if selection['status'] != 'selected':
            raise ValueError('没有候选通过公开筛选；已保存全部拒绝原因，不自动降低门槛')
        return read_json(saved_path)


def product_boxes(record):
    if record.get('boxes'):
        return record['boxes']
    import numpy as np
    from PIL import Image
    foreground = np.asarray(Image.open(record['foreground_mask']).convert('L')) > 127
    ys, xs = np.where(foreground)
    if not len(xs):
        raise ValueError('公开前景掩码为空')
    return [[int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)]]


def generation_definition(selection, options, method='PerLAGE'):
    method = normalize_method(method)
    if method not in NATURAL_METHODS:
        raise ValueError('只有感知生成方法具有四阶段配置')
    from src.visual._perlage.evidence import definition_for_method
    from src.visual.retrieval_objective import TRAINING_DATA_POLICY
    if set(options) - {'harmonization', 'latent', 'initialization', 'pixel_optimization'}:
        raise ValueError('生成参数只能配置初始化、颜色融合和潜变量细化；不能传入历史成图或记忆')
    source, target = selection['source'], selection['target']
    is_no_pixel = method == 'PerLAGE_no_pixel'
    feature_text = selected_feature_text(selection)
    training_queries = [feature_text] if feature_text else list(TRAINING_QUERIES)
    definition = dict(method=method, definition=definition_for_method(method), color_scheme='statistics',
        training_data_policy=TRAINING_DATA_POLICY, pixel_optimization=False if is_no_pixel else options.get('pixel_optimization', False),
        data=dict(source=source['image'], target=target['image'], source_title=source['title'],
                  training_queries=training_queries),
        target_feature_text=feature_text,
        source_boxes=product_boxes(source), target_boxes=product_boxes(target),
        print_regions=source['text_regions'])
    definition.update(copy.deepcopy(options))
    if is_no_pixel:
        definition['pixel_optimization'] = False
        definition.setdefault('latent', {})['auxiliary_levels'] = 0
    return definition


def load_frozen(method_root):
    from src.visual.attack_completion import validate_attack_completion
    method_root = Path(method_root)
    frozen = read_json(method_root / 'FROZEN_IMAGE.json')
    binding = read_json(method_root / 'PUBLIC_RUN.json')
    selection_path = method_root.parents[1] / 'selection/SELECTED_PAIR.json'
    selection = read_json(selection_path)
    if (frozen.get('status') != 'frozen' or frozen['public_binding_sha256'] != digest(binding)
            or binding['selection_sha256'] != sha256_file(selection_path)
            or frozen['source'] != selection['source'] or frozen['target'] != selection['target']
            or frozen['method'] != binding['method']
            or sha256_file(frozen['image']) != frozen['image_sha256']):
        raise ValueError('最终图片、公开图对及冻结方法未形成一致绑定')
    validate_attack_completion(frozen['metadata'], frozen['requested_steps'],
        dict(method=frozen['method'], source_image_sha256=frozen['source']['image_sha256'],
             target_image_sha256=frozen['target']['image_sha256']))
    return frozen


def generate(config, root, selection, method, *, device='cuda', generator=None):
    from src.visual.optimization import generate_attack
    from src.visual.attack_completion import validate_attack_completion
    method = normalize_method(method)
    generation = config.get('generation', {})
    if set(generation) - set(METHODS):
        raise ValueError('生成设置只能使用当前登记的三种方法')
    options = copy.deepcopy(generation.get(method, {}))
    feature_text = selected_feature_text(selection)
    if feature_text is None:
        raise ValueError('正式生成前须先在选图阶段提取并冻结目标商品特点文字')
    if method == 'lucid':
        if options.get('target_text', feature_text) != feature_text:
            raise ValueError('像素方法与感知方法必须使用同一份目标商品特点文字')
        options['target_text'] = feature_text
        options.setdefault('text_weight', 1.0)
    output = Path(root) / 'methods' / method
    source, target = selection['source'], selection['target']
    public_binding = dict(method=method, selection_sha256=sha256_file(Path(root) / 'selection/SELECTED_PAIR.json'),
                          source=source, target=target, options=options, source_files=source_snapshot(),
                          shared_pool_scope='shared', threat_model='black_box_provisional')
    evaluation = config.get('evaluation', {})
    query_mode = evaluation.get('query_mode', 'joint')
    query_request_name = 'QUERY_INPUT_REQUEST.json' if query_mode == 'text' else 'QUERY_IMAGE_REQUEST.json'
    private_paths = [project_path(evaluation.get('shared_pool', SHARED_MEMORY)),
                     project_path(evaluation.get('query_manifest', PRIVATE / 'query_image.json')),
                     project_path(evaluation.get('encoder_config', PRIVATE / 'evaluator.json'))]
    with catalog_training_guard(allowed_output_roots=[output, Path(root) / 'selection',
            Path(source['image']).parent, Path(target['image']).parent], private_paths=private_paths) as audit:
        freeze_json(output / 'PUBLIC_RUN.json', public_binding)
        if (output / 'FROZEN_IMAGE.json').exists():
            frozen = read_json(output / 'FROZEN_IMAGE.json')
            if frozen['public_binding_sha256'] != digest(public_binding):
                raise ValueError('成图冻结后不能更改公开配置或实现')
            validate_attack_completion(frozen['metadata'], frozen['requested_steps'],
                dict(method=method, source_image_sha256=source['image_sha256'], target_image_sha256=target['image_sha256']))
            freeze_json(output / query_request_name, pending_query_spec(target, query_mode))
            return frozen
        if method in NATURAL_METHODS:
            definition = generation_definition(selection, options, method=method)
            freeze_json(output / 'definition.json', definition)
            parameter = 'perlage_no_pixel_config' if method == 'PerLAGE_no_pixel' else 'perlage_config'
            image, metadata = (generator or generate_attack)(source['image'], target['image'], output / 'generation',
                method=method, device=device, **{parameter: definition})
            steps = metadata['perlage_ready']['stages']['latent_generation_refinement']['steps_requested']
        else:
            if set(options) - {'steps', 'epsilon_levels', 'target_text', 'text_weight'}:
                raise ValueError('像素方法生成选项含未知字段')
            steps = options.get('steps', 1000)
            image, metadata = (generator or generate_attack)(source['image'], target['image'], output / 'generation',
                method=method, device=device, **options)
        validate_attack_completion(metadata, steps, dict(method=method,
            source_image_sha256=source['image_sha256'], target_image_sha256=target['image_sha256']))
        final = output / 'frozen.png'
        final.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(image, final)
        frozen = dict(schema_version=1, method=method, status='frozen', image=str(final.resolve()),
                      image_sha256=sha256_file(final), source=source, target=target,
                      metadata=metadata, requested_steps=steps, public_binding_sha256=digest(public_binding),
                      shared_pool_scope='shared', victim_memory_used=False, formal_queries_used=False)
        freeze_json(output / 'FROZEN_IMAGE.json', frozen)
        freeze_json(output / 'TRAINING_ACCESS_AUDIT.json', {k: v for k, v in audit.items() if k != 'active'})
    freeze_json(output / query_request_name, pending_query_spec(target, query_mode))
    return frozen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('select', 'generate', 'evaluate', 'verify', 'report', 'all', 'doctor'), nargs='?', default='all')
    parser.add_argument('--config', type=Path, default=EXPERIMENT_CONFIGS / 'pipeline.example.json')
    parser.add_argument('--method', choices=METHODS)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--query-mode', choices=('text', 'joint'), help='正式询问采用纯文字或独立图文协议')
    parser.add_argument('--query-manifest', type=Path, help='显式复用已冻结的正式询问清单')
    parser.add_argument('--no-defense', action='store_true', help='仅运行无防御条件，不运行完整防御面板')
    args = parser.parse_args(argv)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1')
    if args.stage == 'doctor':
        from src.experiments.runtime_check import doctor
        doctor(args.device)
        return 0
    config = load_config(args.config)
    if args.query_mode is not None:
        config.setdefault('evaluation', {})['query_mode'] = args.query_mode
    if args.query_manifest is not None:
        config.setdefault('evaluation', {})['query_manifest'] = str(args.query_manifest.resolve())
    if args.no_defense:
        config.setdefault('evaluation', {})['defense_mode'] = 'none'
    method = normalize_method(args.method or config.get('method', 'PerLAGE'))
    root = project_path(config.get('run_dir', OUTPUT / 'runs/default'))
    if not root.is_relative_to(OUTPUT.resolve()):
        parser.error('运行结果须置于新项目的输出目录，不得写入原项目')
    root.mkdir(parents=True, exist_ok=True)
    freeze_json(root / 'STOP_SHOPPING_REQUESTED.json', dict(shopping_allowed=False, external_model_calls_allowed=False))
    method_root = root / 'methods' / method
    try:
        if args.stage in ('select', 'generate', 'all'):
            selection = select_pair(config, root, device=args.device)
            if args.stage == 'select':
                print('公开图对筛选已完成：' + str(root / 'selection/SELECTED_PAIR.json'))
                return 0
        if args.stage in ('generate', 'all'):
            generate(config, root, selection, method, device=args.device)
        if args.stage in ('evaluate', 'all'):
            from src.visual.attack_completion import validate_attack_completion
            from src.experiments.shared_evaluation import evaluate, report
            frozen = load_frozen(method_root)
            settings = config.get('evaluation', {})
            if set(settings) - {'shared_pool', 'query_manifest', 'query_mode', 'defense_mode', 'source_memory_id', 'encoder_config'}:
                raise ValueError('评估只接受共享池、正式查询图和评估端记忆标识')
            from src.experiments.defenses import conditions
            selected_conditions = ([c for c in conditions() if c['kind'] == 'none']
                                   if settings.get('defense_mode') == 'none' else None)
            evaluate(frozen, method_root / 'evaluation',
                     pool_path=project_path(settings.get('shared_pool', SHARED_MEMORY)),
                     query_manifest=project_path(settings.get('query_manifest', PRIVATE / 'query_image.json')),
                     source_memory_id=settings.get('source_memory_id'), device=args.device,
                     query_mode=settings.get('query_mode'),
                     selected_conditions=selected_conditions,
                     encoder_config=project_path(settings['encoder_config']) if settings.get('encoder_config') else None)
            report(method_root / 'evaluation')
        if args.stage == 'verify':
            from src.experiments.shared_evaluation import verify
            settings = config.get('evaluation', {})
            from src.experiments.shared_evaluation import make_eval_encoder
            verify(method_root / 'evaluation', encoder_factory=lambda: make_eval_encoder(args.device,
                project_path(settings['encoder_config']) if settings.get('encoder_config') else None),
                query_mode=settings.get('query_mode'),
                query_manifest=project_path(settings['query_manifest']) if settings.get('query_manifest') else None)
        if args.stage == 'report':
            from src.experiments.shared_evaluation import report
            report(method_root / 'evaluation')
    except SelectedAnnotationsPending as error:
        save_json(root/'RUN_STATUS.json',dict(status='awaiting_selected_public_annotations',reason=str(error),generation_completed=False))
        print(str(error))
        return 2
    except QueryImagePending as error:
        save_json(method_root / 'RUN_STATUS.json', dict(status='awaiting_query_image', reason=str(error),
                                                       generation_completed=True, evaluation_completed=False))
        print(str(error))
        return 2
    except Exception as error:
        save_json(method_root / 'RUN_STATUS.json', dict(status='failed', stage=args.stage,
                                                       error_type=type(error).__name__, reason=str(error)))
        raise
    status = ('selected' if args.stage == 'select' else 'generated' if args.stage == 'generate'
              else 'completed' if args.stage in ('all', 'evaluate', 'report', 'verify') else args.stage)
    save_json(method_root / 'RUN_STATUS.json', dict(status=status, method=method,
                                                  shared_pool_scope='shared', stage=args.stage))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
