import json
from pathlib import Path
import pytest

from src.paths import SHARED_MEMORY
from src.vision.optimization import METHODS, normalize_method
from src.vision.catalog_training_guard import catalog_training_guard
from src.vision.audit import save_json, read_json
from src.vision.image_utils import sha256_file
from src.experiments.query_contract import load_formal_queries, QueryImagePending
from src.experiments.formal_queries import VAGUE_INSTRUCTIONS
from src.experiments.pipeline import TRAINING_QUERIES, generation_definition
from src.experiments.defenses import conditions, table_groups
from src.vision._perlage.pipeline import validate_definition


def test_method_names_are_unambiguous():
    assert METHODS == ('lucid', 'PerLAGE')
    assert normalize_method('perlage') == 'PerLAGE'
    for old in ('nf_lage', 'nf_lage_opt', 'nf_lage_no_shadow', 'legacy_pixel'):
        with pytest.raises(ValueError): normalize_method(old)


def test_training_cannot_read_configured_private_files(tmp_path):
    secret = tmp_path / 'unusual_name.json'
    secret.write_text('私有记录', encoding='utf8')
    output = tmp_path / 'public_run'
    with catalog_training_guard(allowed_output_roots=[output], private_paths=[secret]) as state:
        with pytest.raises(PermissionError): secret.read_text(encoding='utf8')
        output.mkdir()
        (output / 'public_identity.json').write_text('{}', encoding='utf8')
        with pytest.raises(PermissionError): (output / 'history.json').write_text('{}', encoding='utf8')
    assert len(state['blocked_accesses']) == 2
    assert secret.read_text(encoding='utf8') == '私有记录'


def test_training_snapshot_excludes_formal_requests_and_guard_blocks_them(tmp_path):
    from src.paths import PROJECT
    from src.experiments.pipeline import source_snapshot
    formal = PROJECT / 'src/experiments/formal_queries.py'
    assert 'src/experiments/formal_queries.py' not in source_snapshot()
    with catalog_training_guard(allowed_output_roots=[tmp_path]):
        with pytest.raises(PermissionError):
            formal.read_text(encoding='utf8')


def query_manifest(products, tmp_path, **overrides):
    path = tmp_path / 'query.json'
    document = dict(status='ready', image=products[2]['image'], sha256=products[2]['image_sha256'],
                    target_public_id=products[0]['id'],
                    review=dict(no_explicit_product_name=True, no_brand=True, reviewer='人工夹具'))
    document.update(overrides)
    save_json(path, document)
    return path


def test_query_image_is_separate_and_reviewed(products, tmp_path):
    with pytest.raises(QueryImagePending): load_formal_queries(tmp_path/'missing.json', products[0])
    queries = load_formal_queries(query_manifest(products, tmp_path), products[0])
    assert len(queries) == 20 and len({q['instruction'] for q in queries}) == 20
    assert not set(VAGUE_INSTRUCTIONS) & set(TRAINING_QUERIES)
    with pytest.raises(ValueError):
        load_formal_queries(query_manifest(products,tmp_path,image=products[0]['image'],sha256=products[0]['image_sha256']),products[0])
    with pytest.raises(QueryImagePending):
        load_formal_queries(query_manifest(products,tmp_path,review={}),products[0])


def test_complete_definition_keeps_original_numeric_defaults(products):
    definition = generation_definition(dict(source=products[1], target=products[0]), {})
    config = validate_definition(definition)
    assert definition['method'] == 'PerLAGE'
    assert config.latent['steps'] == 60
    assert config.latent['source_ssim_floor'] == .85
    assert config.latent['source_mae_ceiling'] == 12
    assert config.latent['guard_backtracks'] == 8
    assert 'public_pool' not in definition and 'source_history' not in definition['data']
    partial = generation_definition(dict(source=products[1], target=products[0]), {'latent': {'steps': 20}})
    assert validate_definition(partial).latent['ordinary_tolerance'] == .01


def test_fixed_defense_panel_preserves_all_seeds():
    specs = conditions()
    assert len(specs) == 26 and len(table_groups()) == 14
    assert len({s['label'] for s in specs}) == 26
    for timestep in (99,199,399):
        assert {s['seed'] for s in specs if s.get('timestep') == timestep} == set(range(1001,1006))


def test_evaluation_recomputes_rankings_with_fresh_encoder(pool_inputs, products, tmp_path):
    from src.memory.shared_pool import build_shared_pool
    from src.experiments.shared_evaluation import evaluate, report, verify
    from test_shared_pool import ToyEncoder
    pool, _ = pool_inputs
    shared = tmp_path/'shared_memory.json'
    build_shared_pool([pool], shared, kind='purchase_memory')
    frozen = dict(method='lucid', status='frozen', source=products[1], target=products[0],
                  image=products[2]['image'], image_sha256=products[2]['image_sha256'], requested_steps=2,
                  metadata=dict(method='lucid',status='success',output_image=products[2]['image'],
                                output_image_sha256=products[2]['image_sha256'],
                                attack=dict(steps_completed=2,steps_requested=2)))
    created = []
    def factory():
        encoder = ToyEncoder(); created.append(encoder); return encoder
    result = evaluate(frozen,tmp_path/'evaluation', pool_path=shared,
                      query_manifest=query_manifest(products,tmp_path), encoder_factory=factory,
                      selected_conditions=[conditions()[0]])
    assert len(created) == 2 and all(e.closed for e in created)
    assert result['verification']['rankings_recomputed'] == 40
    assert result['verification']['max_score_error'] == 0
    assert report(tmp_path/'evaluation').is_file()
    rows = read_json(tmp_path/'evaluation/evaluation.json')
    rows[0]['queries'][0]['score'] = float('nan')
    # 写入故障数据以验证复算拒绝非有限分数。
    (tmp_path/'evaluation/evaluation.json').write_text(json.dumps(rows),encoding='utf8')
    with pytest.raises(ValueError): verify(tmp_path/'evaluation',encoder_factory=factory)
