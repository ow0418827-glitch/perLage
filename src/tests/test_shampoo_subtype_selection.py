import copy
from src.paths import ProjectPath as Path
import numpy as np
from PIL import Image
import pytest

from src.experiments import public_pair_selection as flow
from src.visual.audit import digest, read_json, save_json
from src.visual.image_utils import sha256_file
from src.visual.pair_assets import load_pair
from src.visual.pair_selection import choose_pair, matches_subtype_filter, score_pair, verify_selection
from src.visual.shape_calibration import calibrate, load_calibration
from src.visual.selection.webshop_public import extract_public_candidates, simulated_controlled


SHAMPOO_RULE = {
    'policy': 'shampoo_human_hair_only_v1',
    'include': r'\bshampoos?\b|洗发',
    'exclude': r'\b(?:carpet|rug|upholstery|auto|car|pet|dog|cat|puppy|horse|equine|floor|machine|extracter|extractor|empty|wig|doll|cape|basin|station)\b|\bshampoo\s+(?:brush|bowl|sink|mat|tray)s?\b',
}


def _public_item(pid, title):
    return dict(asin=pid, name=title, images=['https://m.media-amazon.com/images/I/' + pid + '.jpg'])


class _ShapeBackend:
    snapshot = {'fixture': True}

    def __init__(self, *args, **kwargs):
        pass

    def features(self, product):
        return {'shape': [1.0, 0.0]}

    def close(self):
        pass


def test_matches_subtype_filter_handles_include_exclude_and_empty():
    assert matches_subtype_filter('Dove Hand Wash 16 oz', None)
    assert matches_subtype_filter('Dove Hand Wash 16 oz', {})
    assert matches_subtype_filter('Aveda Scalp Benefits Balancing Shampoo 33.8 OZ', SHAMPOO_RULE)
    assert matches_subtype_filter('Herbal Essences Bio:Renew Hydrating Shampoos', SHAMPOO_RULE)
    assert not matches_subtype_filter('Dove Relaxing Lavender Hand Wash 16.9 oz', SHAMPOO_RULE)
    assert not matches_subtype_filter('Aloe Vera Hand & Body Lotion 16 oz', SHAMPOO_RULE)
    assert not matches_subtype_filter('Bissel DeepClean Carpet Shampoo 60 oz', SHAMPOO_RULE)
    assert not matches_subtype_filter('Oatmeal Natural Pet Dog Shampoo 16 oz', SHAMPOO_RULE)
    assert not matches_subtype_filter('Silicone Scalp Massager Shampoo Brush', SHAMPOO_RULE)


def test_choose_pair_enforces_subtype_filter_on_target_and_candidates(products, calibration_factory, tmp_path):
    cfg = calibration_factory()
    target = dict(products[0], title='Aveda Scalp Benefits Balancing Shampoo 33.8 OZ')
    lotion = dict(products[1], id='lotion0001', title='Dove Refreshing Hand & Body Lotion 16.9 oz')
    pet_shampoo = dict(products[1], id='petshamp01', title='Gentle Dog Pet Shampoo 16 oz')
    img_pet = tmp_path / 'pet.png'
    Image.new('RGB', (24, 24), (70, 90, 110)).save(img_pet)
    pet_shampoo.update(image=str(img_pet), image_sha256=sha256_file(img_pet))

    valid_shampoo = dict(products[2], id='shampoo001', owned=True, title='Daily Moisturizing Hair Shampoo 32 oz')

    # Incompatible target fails fast.
    bad_target = dict(products[0], title='Luxury Body Lotion 16 oz')
    with pytest.raises(ValueError, match='细分品类规则'):
        choose_pair(bad_target, [valid_shampoo], _ShapeBackend(), cfg, subtype_filter=SHAMPOO_RULE)

    row_lotion = score_pair(
        lotion, target, {'shape': [1.0, 0.0]}, {'shape': [1.0, 0.0]},
        threshold=0.5, config=cfg, subtype_filter=SHAMPOO_RULE,
    )
    assert 'subtype_filter_mismatch' in row_lotion['rejected_reasons'] and not row_lotion['eligible']

    result = choose_pair(target, [lotion, pet_shampoo, valid_shampoo], _ShapeBackend(), cfg, subtype_filter=SHAMPOO_RULE)
    by_id = {row['source_id']: row for row in result['candidates']}
    assert 'subtype_filter_mismatch' in by_id['lotion0001']['rejected_reasons']
    assert 'subtype_filter_mismatch' in by_id['petshamp01']['rejected_reasons']
    assert by_id['shampoo001']['eligible'] is True
    assert result['selected_source_id'] == 'shampoo001'
    result.update(target=target, source=valid_shampoo)
    verify_selection(result)


def test_extract_public_candidates_pre_filters_before_shortlist_and_locks_identity(tmp_path):
    target = 'T000000001'
    controlled = [f'C{i:09d}' for i in range(60) if simulated_controlled(f'C{i:09d}', 42, {target})]
    rows = [
        _public_item(target, 'Target Botanical Balancing Shampoo 16 oz'),
        _public_item(controlled[0], 'Lavender Shower Gel 16 oz'),
        _public_item(controlled[1], 'Aloe Body Lotion 16 oz'),
        _public_item(controlled[2], 'Heavy Duty Carpet Shampoo Cleaner'),
        _public_item(controlled[3], 'Tear Free Puppy Dog Shampoo'),
        _public_item(controlled[4], 'Daily Hydrating Hair Shampoo 16 oz'),
        _public_item(controlled[5], 'Clarifying Herbal Hair Shampoo 12 oz'),
    ]
    products_path = tmp_path / 'products.json'
    save_json(products_path, rows)
    groups = [dict(key='07_shampoo', target_id=target)]
    filters = {'07_shampoo': SHAMPOO_RULE}

    extracted = extract_public_candidates(
        products_path, groups, tmp_path / 'public', lexical_limit=32, subtype_filters=filters
    )
    group = extracted['groups'][0]
    assert [c['id'] for c in group['candidates']] == sorted([controlled[4], controlled[5]],
        key=lambda pid: (digest(['public_candidate_uniform_v2', 42, target, pid]), pid))
    assert group['counts']['subtype_filtered'] == 4
    assert group['counts']['type_compatible'] == 3  # target + 2 human shampoos

    # Same directory cannot silently reuse cached shortlist under changed subtype_filters.
    with pytest.raises(ValueError, match='公开候选规则改变'):
        extract_public_candidates(
            products_path, groups, tmp_path / 'public', lexical_limit=32, subtype_filters=None
        )

    # Target failing its own subtype filter stops immediately.
    bad_products = tmp_path / 'bad_products.json'
    save_json(bad_products, [_public_item(target, 'Lavender Shower Gel 16 oz'), rows[5]])
    with pytest.raises(ValueError, match='细分品类规则'):
        extract_public_candidates(
            bad_products,
            groups,
            tmp_path / 'bad_public/out',
            lexical_limit=32,
            index_path=tmp_path / 'bad_public/index.sqlite',
            subtype_filters=filters,
        )


def test_validate_config_checks_subtype_filters():
    base = dict(
        schema_version=2,
        seed=42,
        candidate_limit=32,
        calibration_path='data/cal.json',
        groups=[dict(key='07_shampoo', target_id='B00BDJODWI')],
    )
    valid = dict(base, subtype_filters={'07_shampoo': SHAMPOO_RULE})
    assert flow.validate_config(valid) == valid

    for bad_filters in (
        'bad_not_dict',
        {'02_chair': SHAMPOO_RULE},
        {'07_shampoo': None},
        {'07_shampoo': {}},
        {'07_shampoo': {'policy': 'shampoo_human_hair_only_v1'}},
        {'07_shampoo': {'unknown': 'x'}},
        {'07_shampoo': {'policy': ''}},
        {'07_shampoo': {'include': ''}},
        {'07_shampoo': {'include': '[unclosed'}},
    ):
        with pytest.raises(ValueError):
            flow.validate_config(dict(base, subtype_filters=bad_filters))


def test_subtype_filter_rejects_noop_typo_and_non_dict_across_apis(products, calibration_factory, tmp_path):
    cfg = calibration_factory()
    target = dict(products[0], title='Aveda Scalp Benefits Balancing Shampoo 33.8 OZ')
    valid_shampoo = dict(products[1], owned=True, title='Daily Moisturizing Hair Shampoo 32 oz')

    for bad in ('shampoo', ['shampoo'], {'incldue': r'\bshampoo\b'}, {'policy': 'only_policy'}, {'policy': ''}, {'include': ''}, {'include': '[bad'}):
        with pytest.raises(ValueError):
            matches_subtype_filter('Daily Hair Shampoo', bad)

    for bad_api_rule in ({}, {'policy': 'only_policy'}, {'incldue': r'\bshampoo\b'}, {'include': '[bad'}):
        with pytest.raises(ValueError):
            score_pair(valid_shampoo, target, {'shape': [1.0, 0.0]}, {'shape': [1.0, 0.0]},
                       threshold=0.5, config=cfg, subtype_filter=bad_api_rule)
        with pytest.raises(ValueError):
            choose_pair(target, [valid_shampoo], _ShapeBackend(), cfg, subtype_filter=bad_api_rule)

    target_id = 'T000000001'
    products_path = tmp_path / 'p.json'
    save_json(products_path, [_public_item(target_id, 'Target Hair Shampoo')])
    groups = [dict(key='07_shampoo', target_id=target_id)]
    for idx, bad_filters in enumerate((
        'not_a_dict',
        {'wrong_group': SHAMPOO_RULE},
        {'07_shampoo': None},
        {'07_shampoo': {}},
        {'07_shampoo': {'policy': 'noop'}},
        {'07_shampoo': {'incldue': r'\bshampoo\b'}},
    )):
        with pytest.raises(ValueError):
            extract_public_candidates(
                products_path, groups, tmp_path / f'out_{idx}',
                index_path=tmp_path / 'idx.sqlite', subtype_filters=bad_filters,
            )


def test_calibrate_supports_isolated_dev_manifest_with_dual_public_root(tmp_path, monkeypatch):
    import src.visual.pair_selection as ps_mod
    import src.experiments.public_pair_selection as flow_mod

    public = tmp_path / 'datasets/public'
    (public / 'images').mkdir(parents=True)
    dev_records = []
    for i in range(4):
        img = public / 'images' / f'dev{i}.png'
        Image.new('RGB', (80, 80), (30 + i * 25, 90, 140)).save(img)
        dev_records.append(dict(id=f'dev_{i}', title=f'Public Shampoo {i}', image=str(img), source_note='合成开发集'))

    root = tmp_path / 'output/public_pair_selection/iso_cal_test'
    working = root / 'data'
    (working / 'dev').mkdir(parents=True)
    dev_manifest = working / 'dev/development.json'
    save_json(dev_manifest, dict(
        schema_version=1,
        reviewer='合成核对',
        pairs=[
            dict(left=dev_records[0], right=dev_records[1], label=1, review_note='正对'),
            dict(left=dev_records[2], right=dev_records[3], label=0, review_note='负对'),
        ],
    ))

    class DevBackend(_ShapeBackend):
        def features(self, product):
            vec = {
                'dev_0': [1.0, 0.0],
                'dev_1': [0.92, float(np.sqrt(1 - 0.92**2))],
                'dev_2': [1.0, 0.0],
                'dev_3': [0.18, float(np.sqrt(1 - 0.18**2))],
            }[product['id']]
            return {'shape': vec}

    monkeypatch.setattr(flow_mod, 'PUBLIC', public)
    monkeypatch.setattr(flow_mod, 'OUTPUT', tmp_path / 'output')
    monkeypatch.setattr(flow_mod, 'LocalSelectionBackend', DevBackend)

    cert = working / 'calibration/threshold.json'
    cfg_path = working / 'selection_config.json'
    save_json(cfg_path, dict(
        schema_version=2, seed=42, candidate_limit=32, calibration_path=str(cert),
        groups=[dict(key='07_shampoo', target_id='B00BDJODWI')],
        subtype_filters={'07_shampoo': SHAMPOO_RULE},
    ))
    assert flow.main([
        'calibrate', '--config', str(cfg_path), '--development', str(dev_manifest),
        '--root', str(root), '--device', 'cpu',
    ]) == 0
    loaded = load_calibration(cert, DevBackend.snapshot, public_root=(working, public))
    assert 0.18 < loaded['fit']['threshold'] < 0.92

    # Outside image still rejected BEFORE reading image file bytes.
    outside_img = tmp_path / 'outside.png'
    Image.new('RGB', (80, 80), (1, 2, 3)).save(outside_img)
    bad_manifest = working / 'dev/bad_development.json'
    bad_pair_left = dict(dev_records[0], image=str(outside_img))
    save_json(bad_manifest, dict(
        schema_version=1, reviewer='合成核对',
        pairs=[
            dict(left=bad_pair_left, right=dev_records[1], label=1, review_note='正对'),
            dict(left=dev_records[2], right=dev_records[3], label=0, review_note='负对'),
        ],
    ))
    monkeypatch.setattr(ps_mod, 'sha256_file', lambda *a, **k: (_ for _ in ()).throw(AssertionError('不得先读取越界文件')))
    with pytest.raises(ValueError, match='公开数据目录'):
        calibrate(bad_manifest, working / 'calibration/bad.json', DevBackend(), public_root=(working, public))


def test_pipeline_select_pair_handles_isolated_catalog_and_isolated_calibration(tmp_path, monkeypatch, public_feature_stub):
    from src.experiments import pipeline

    public = tmp_path / 'output/datasets/public'
    private = tmp_path / 'output/datasets/private'
    data_dir = tmp_path / 'data'
    data_dir.mkdir(parents=True)
    dev_dir = public / 'calibration/dev'
    dev_dir.mkdir(parents=True)
    dev_records = []
    for i in range(4):
        img = dev_dir / f'dev{i}.png'
        Image.new('RGB', (80, 80), (20 + i * 25, 80, 120)).save(img)
        dev_records.append(dict(id=f'dev_{i}', title=f'Public Dev Shampoo {i}', image=str(img), source_note='合成开发集'))
    dev_manifest = dev_dir / 'development.json'
    save_json(dev_manifest, dict(
        schema_version=1, reviewer='合成核对',
        pairs=[
            dict(left=dev_records[0], right=dev_records[1], label=1, review_note='正对'),
            dict(left=dev_records[2], right=dev_records[3], label=0, review_note='负对'),
        ],
    ))

    iso_data = tmp_path / 'output/public_pair_selection/iso_run/data'
    (iso_data / 'images').mkdir(parents=True)
    cert = iso_data / 'calibration/threshold.json'

    class DevBackend(_ShapeBackend):
        def features(self, product):
            vec = {
                'dev_0': [1.0, 0.0],
                'dev_1': [0.95, float(np.sqrt(1 - 0.95**2))],
                'dev_2': [1.0, 0.0],
                'dev_3': [0.15, float(np.sqrt(1 - 0.15**2))],
            }.get(product['id'], [1.0, 0.0])
            return {'shape': vec}

    calibrate(dev_manifest, cert, DevBackend(), public_root=public)

    src_img = iso_data / 'images/src.png'
    tgt_img = iso_data / 'images/tgt.png'
    Image.new('RGB', (80, 80), (40, 100, 160)).save(src_img)
    Image.new('RGB', (80, 80), (160, 100, 40)).save(tgt_img)
    target = dict(
        id='B00BDJODWI', title='Aveda Scalp Benefits Balancing Shampoo 33.8 OZ',
        image=str(tgt_img), source_url='https://www.amazon.com/dp/B00BDJODWI',
        boxes=[[0, 0, 80, 80]], text_regions=[], text_regions_reviewed=True,
    )
    cand = dict(
        id='B09B63YYCW', title='Bellisso Biotin Hair Shampoo 16 OZ',
        image=str(src_img), source_url='https://www.amazon.com/dp/B09B63YYCW',
        owned=True, source_note='模拟可控测试商品',
        boxes=[[0, 0, 80, 80]], text_regions=[], text_regions_reviewed=True,
    )
    catalog = iso_data / 'candidate_catalogs/07_shampoo.json'
    save_json(catalog, dict(schema_version=2, target=target, candidates=[cand]))

    monkeypatch.setattr(pipeline, 'PROJECT', tmp_path)
    monkeypatch.setattr(pipeline, 'PUBLIC', public)
    monkeypatch.setattr(pipeline, 'PRIVATE', private)
    monkeypatch.setattr(pipeline, 'DATA', data_dir)
    monkeypatch.setattr(pipeline, 'SHARED_MEMORY', tmp_path / 'output/shared_memory/pool/shared_memory.json')
    import src.paths as paths_mod
    import src.visual.catalog_training_guard as guard_mod
    monkeypatch.setattr(paths_mod, 'OUTPUT', tmp_path / 'output')
    monkeypatch.setattr(paths_mod, 'PUBLIC', public)
    monkeypatch.setattr(paths_mod, 'DATA', data_dir)
    monkeypatch.setattr(guard_mod, 'OUTPUT', tmp_path / 'output')
    monkeypatch.setattr(guard_mod, 'PRIVATE', private)
    monkeypatch.setattr(guard_mod, 'SHARED_MEMORY', tmp_path / 'output/shared_memory/pool/shared_memory.json')

    run_root = tmp_path / 'output/runs/shampoo_pipeline_test'
    selected = pipeline.select_pair(dict(
        public_catalog=str(catalog),
        selection=dict(seed=42, calibration_path=str(cert), min_image_side=64),
    ), run_root, device='cpu', backend_factory=DevBackend)
    assert selected['status'] == 'selected' and selected['source']['id'] == 'B09B63YYCW'

    # Calibration certificate in a separate isolated data dir works; outside dev manifest is still rejected.
    cal_iso2 = tmp_path / 'output/public_pair_selection/iso_cal2/data'
    cert2 = cal_iso2 / 'calibration/threshold.json'
    calibrate(dev_manifest, cert2, DevBackend(), public_root=public)
    run_root2 = tmp_path / 'output/runs/shampoo_pipeline_test2'
    selected2 = pipeline.select_pair(dict(
        public_catalog=str(catalog),
        selection=dict(seed=42, calibration_path=str(cert2), min_image_side=64),
    ), run_root2, device='cpu', backend_factory=DevBackend)
    assert selected2['source']['id'] == 'B09B63YYCW'

    outside_cert = tmp_path / 'outside_cert.json'
    outside_cert.write_bytes(cert.read_bytes())
    with pytest.raises(ValueError, match='公开输入必须位于'):
        pipeline.select_pair(dict(
            public_catalog=str(catalog),
            selection=dict(seed=42, calibration_path=str(outside_cert), min_image_side=64),
        ), tmp_path / 'output/runs/bad_run', device='cpu', backend_factory=DevBackend)


def test_isolated_calibration_certificate_works_with_shared_public_development_manifest(
    products, tmp_path, monkeypatch, public_feature_stub
):
    public = tmp_path / 'datasets/public'
    dev_dir = public / 'calibration/dev'
    dev_dir.mkdir(parents=True)
    dev_records = []
    for i in range(4):
        img = dev_dir / f'dev{i}.png'
        Image.new('RGB', (80, 80), (20 + i * 25, 60 + i * 10, 110 + i * 5)).save(img)
        dev_records.append(dict(id=f'dev_item_{i}', title=f'Public Shampoo Dev {i}', image=str(img), source_note='合成开发集'))
    dev_manifest = dev_dir / 'development.json'
    save_json(
        dev_manifest,
        dict(
            schema_version=1,
            reviewer='合成测试核对者',
            pairs=[
                dict(left=dev_records[0], right=dev_records[1], label=1, review_note='合成正对'),
                dict(left=dev_records[2], right=dev_records[3], label=0, review_note='合成负对'),
            ],
        ),
    )

    root = tmp_path / 'output/public_pair_selection/shampoo_reselect_test'
    working = root / 'data'
    working.mkdir(parents=True)
    isolated_cert = working / 'calibration/threshold.json'

    class DevBackend(_ShapeBackend):
        def features(self, product):
            vec = {
                'dev_item_0': [1.0, 0.0],
                'dev_item_1': [0.95, float(np.sqrt(1 - 0.95**2))],
                'dev_item_2': [1.0, 0.0],
                'dev_item_3': [0.15, float(np.sqrt(1 - 0.15**2))],
            }.get(product['id'], [1.0, 0.0])
            return {'shape': vec}

    calibrate(dev_manifest, isolated_cert, DevBackend(), public_root=public)
    assert isolated_cert.is_file()

    # Single-root check against `working` rejects `development.json` under `public`; dual-root accepts both cleanly.
    with pytest.raises(ValueError, match='公开目录'):
        load_calibration(isolated_cert, DevBackend.snapshot, public_root=working)
    loaded = load_calibration(isolated_cert, DevBackend.snapshot, public_root=(working, public))
    assert loaded['fit']['threshold'] < 0.95

    # Also rejects paths completely outside both allowed roots.
    outside_cert = tmp_path / 'outside_threshold.json'
    outside_cert.write_bytes(isolated_cert.read_bytes())
    with pytest.raises(ValueError, match='公开目录'):
        load_calibration(outside_cert, DevBackend.snapshot, public_root=(working, public))

    monkeypatch.setattr(flow, 'PUBLIC', public)
    monkeypatch.setattr(flow, 'OUTPUT', tmp_path / 'output')

    target_id = 'B000000001'
    controlled = [f'S{i:09d}' for i in range(60) if simulated_controlled(f'S{i:09d}', 42, {target_id})]
    raw_items = [
        _public_item(target_id, 'Botanical Balancing Hair Shampoo 33.8 OZ'),
        _public_item(controlled[0], 'Creamy Lavender Shower Gel 16.9 OZ'),
        _public_item(controlled[1], 'Soothing Hand & Body Lotion 12 OZ'),
        _public_item(controlled[2], 'Clean Scalp Herbal Shampoo 25 OZ'),
    ]
    products_path = tmp_path / 'raw_products.json'
    save_json(products_path, raw_items)

    config = dict(
        schema_version=2,
        seed=42,
        candidate_limit=32,
        calibration_path=str(isolated_cert),
        groups=[dict(key='07_shampoo', target_id=target_id)],
        subtype_filters={'07_shampoo': SHAMPOO_RULE},
    )
    flow.freeze_plan(config, root)

    extracted = extract_public_candidates(
        products_path,
        config['groups'],
        working,
        seed=config['seed'],
        lexical_limit=config['candidate_limit'],
        index_path=public / 'public_products_v2.sqlite',
        excluded_ids=loaded['development_product_ids'],
        subtype_filters=config['subtype_filters'],
    )
    assert [c['id'] for c in extracted['groups'][0]['candidates']] == [controlled[2]]

    short_products = [extracted['groups'][0]['target']] + extracted['groups'][0]['candidates']
    downloads = {}
    for index, p in enumerate(short_products):
        img = working / 'images' / f'{p["id"]}.png'
        img.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (80, 80), (50 + index * 40, 120, 180)).save(img)
        downloads[p['id']] = dict(p, image=str(img), image_sha256=sha256_file(img))
    short = dict(schema_version=2, groups=[dict(key='07_shampoo', target=short_products[0], candidates=short_products[1:])])
    save_json(working / 'candidate_shortlist.json', short)
    save_json(working / 'download_manifest.json', dict(shortlist_sha256=digest(short), records=downloads, failures={}))

    bundle = flow.select_groups(config, root, working, device='cpu', backend_factory=DevBackend)
    assert bundle['status'] == 'selected'
    assert bundle['groups'][0]['source']['id'] == controlled[2]

    import src.visual.public_annotations as annotation

    def propose(sub_downloads, output):
        proposals = {}
        for pid, p in sub_downloads['records'].items():
            mask = Path(output) / f'{pid}.png'
            mask.parent.mkdir(parents=True, exist_ok=True)
            Image.new('L', (80, 80), 255).save(mask)
            proposals[pid] = dict(boxes=[[0, 0, 80, 80]], foreground_mask=str(mask), text_regions=[])
        document = dict(proposals=proposals)
        save_json(Path(output) / 'annotation_manifest.json', document)
        return document

    monkeypatch.setattr(annotation, 'annotate_downloads', propose)
    monkeypatch.setattr(annotation, 'contact_sheets', lambda *a, **k: None)
    proposed = flow.annotate_selected(root, working)
    save_json(
        working / 'selected_visual_reviews.json',
        dict(
            reviewer='合成测试核对',
            annotation_manifest_sha256=digest(proposed),
            records={
                pid: dict(
                    accepted=True,
                    image_sha256=downloads[pid]['image_sha256'],
                    proposal_sha256=digest(proposal),
                )
                for pid, proposal in proposed['proposals'].items()
            },
        ),
    )
    published = flow.publish_groups(root, working)
    assert len(published['groups']) == 1
    pair = load_pair(root / 'data/groups/07_shampoo/pair.json', working)
    assert pair['source']['id'] == controlled[2]
    assert pair['target']['id'] == target_id

    # Also test end-to-end CLI flow.main prepare + select wiring with subtype_filters
    cli_root = tmp_path / 'output/public_pair_selection/shampoo_cli_test'
    cli_config_path = public / 'cli_selection_config.json'
    save_json(cli_config_path, dict(config, groups=[dict(key='07_shampoo', target_id=target_id)]))
    import src.experiments.public_pair_selection as flow_mod
    monkeypatch.setattr(flow_mod, 'download_shortlist', lambda shortlist, out: save_json(Path(out) / 'download_manifest.json', dict(
        shortlist_sha256=digest(shortlist),
        records={
            target_id: downloads[target_id],
            controlled[2]: downloads[controlled[2]],
        },
        failures={},
    )))
    assert flow.main([
        'prepare',
        '--config', str(cli_config_path),
        '--products', str(products_path),
        '--index', str(public / 'public_products_v2.sqlite'),
        '--root', str(cli_root),
        '--device', 'cpu',
    ]) == 0
    cli_short = read_json(cli_root / 'data/candidate_shortlist.json')
    assert [c['id'] for c in cli_short['groups'][0]['candidates']] == [controlled[2]]

    orig_select_groups = flow_mod.select_groups
    monkeypatch.setattr(
        flow_mod,
        'select_groups',
        lambda cfg, r, p, device='cpu': orig_select_groups(cfg, r, p, device=device, backend_factory=DevBackend),
    )
    assert flow.main([
        'select',
        '--config', str(cli_config_path),
        '--root', str(cli_root),
        '--device', 'cpu',
    ]) == 0
    cli_bundle = read_json(cli_root / 'PAIR_SELECTION_BUNDLE.json')
    assert cli_bundle['status'] == 'selected' and cli_bundle['groups'][0]['source']['id'] == controlled[2]
    for stage in ('prepare', 'select'):
        assert read_json(cli_root / f'ACCESS_AUDIT_{stage}.json')['blocked_accesses'] == []

