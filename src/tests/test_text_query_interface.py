"""可复用纯文字询问接口的合成夹具验证，不执行正式实验。"""
import copy
from types import SimpleNamespace

import pytest

from src.visual.audit import read_json, save_json
from src.visual.image_utils import sha256_file
from src.experiments.formal_queries import build_feature_text_manifest, main as query_main
from src.experiments.query_contract import load_formal_queries, QueryImagePending, verify_query_input


@pytest.fixture
def features():
    return dict(asin='fixture_target', category='camera',
                core_attributes=['optical zoom', 'motion detection', 'night vision', 'waterproof'],
                attributes=['optical zoom', 'motion detection', 'night vision', 'waterproof', 'irrelevant extra'],
                canonical_sentence='I want to buy a camera with optical zoom and motion detection.')


def test_fixed_twenty_feature_intents_and_overlap_report(features):
    manifest = build_feature_text_manifest(features)
    assert manifest == build_feature_text_manifest(features)
    assert manifest['query_mode'] == 'text' and len(set(manifest['instructions'])) == 20
    assert manifest['instructions'][:3] == [
        'I want to buy a camera with optical zoom and motion detection.',
        'I would like to purchase a camera featuring optical zoom and night vision.',
        'Looking to buy a camera with motion detection and night vision.']
    assert all('waterproof' not in text and 'irrelevant extra' not in text for text in manifest['instructions'])
    assert manifest['training_overlap']['normalized_exact_overlap_count'] == 1
    assert manifest['training_overlap']['query_numbers'] == [1]
    assert manifest['query_image_required'] is False


@pytest.mark.parametrize('nested', [False, True])
def test_preparation_cli_accepts_direct_or_package_features(features, tmp_path, nested):
    path = tmp_path/'features.json'
    save_json(path, dict(target=features) if nested else features)
    output = tmp_path/'queries.json'
    args = ['--mode', 'feature-text', '--features', str(path), '--output', str(output)]
    assert query_main(args) == 0 and query_main(args) == 0
    assert read_json(output) == build_feature_text_manifest(features)


def test_text_queries_never_read_any_image(features, tmp_path, monkeypatch):
    import src.experiments.query_contract as contract
    path = tmp_path/'queries.json'
    save_json(path, build_feature_text_manifest(features))
    def forbidden(path):
        raise AssertionError('纯文字模式访问了图片摘要')
    monkeypatch.setattr(contract, 'sha256_file', forbidden)
    queries = load_formal_queries(path, dict(id=features['asin']), query_mode='text')
    assert len(queries) == 20 and all(q['query_image'] is None for q in queries)
    for query in queries:
        verify_query_input(query)
    with pytest.raises(ValueError, match='模式与冻结清单不一致'):
        load_formal_queries(path, dict(id=features['asin']), query_mode='joint')
    with pytest.raises(ValueError, match='未绑定当前公开目标商品'):
        load_formal_queries(path, dict(id='wrong_group_target'), query_mode='text')


def test_group_cli_dynamically_loads_only_selected_group_features(features, tmp_path):
    other = dict(features, asin='another_target', category='shampoo',
                 core_attributes=['gentle cleansing', 'easy rinsing', 'moisturizing ingredients'])
    groups = {}
    for key, data in [('camera_group', features), ('shampoo_group', other)]:
        path = tmp_path/'groups'/key/'features.json'
        save_json(path, dict(target=data))
        groups[key] = dict(target_id=data['asin'], features=path.relative_to(tmp_path).as_posix(),
                           features_sha256=sha256_file(path))
    save_json(tmp_path/'manifest.json', dict(groups=groups))
    for key in groups:
        output = tmp_path/(key+'_queries.json')
        assert query_main(['--mode', 'feature-text', '--base-package', str(tmp_path),
                           '--group', key, '--output', str(output)]) == 0
        manifest = read_json(output)
        assert manifest['target_public_id'] == groups[key]['target_id']
    first = read_json(tmp_path/'camera_group_queries.json')
    second = read_json(tmp_path/'shampoo_group_queries.json')
    assert 'optical zoom' in first['instructions'][0]
    assert 'gentle cleansing' in second['instructions'][0]
    assert all('optical zoom' not in text for text in second['instructions'])


@pytest.mark.parametrize('change', ['missing', 'nineteen', 'duplicate', 'blank', 'changed_digest', 'attached_image'])
def test_text_queries_cannot_fall_back_or_accept_malformed_inputs(features, tmp_path, change):
    manifest = build_feature_text_manifest(features)
    if change == 'missing':
        manifest.pop('instructions')
    elif change == 'nineteen':
        manifest['instructions'].pop()
    elif change == 'duplicate':
        manifest['instructions'][1] = manifest['instructions'][0]
    elif change == 'blank':
        manifest['instructions'][1] = '  '
    elif change == 'changed_digest':
        manifest['instructions'][0] += ' Changed.'
    else:
        manifest['image'] = 'not_allowed.png'
    path = tmp_path/'queries.json'
    save_json(path, manifest)
    with pytest.raises(ValueError):
        load_formal_queries(path, dict(id=features['asin']))


def test_joint_mode_still_waits_for_independent_image(features, tmp_path):
    manifest = build_feature_text_manifest(features)
    manifest.update(query_mode='joint', review=dict(no_explicit_product_name=True, no_brand=True, reviewer='合成核对'))
    path = tmp_path/'queries.json'
    save_json(path, manifest)
    with pytest.raises(QueryImagePending):
        load_formal_queries(path, dict(id=features['asin']), query_mode='joint')


def test_text_evaluation_and_fresh_verification_with_no_query_images(pool_inputs, products, tmp_path):
    from src.agent.memory.shared_pool import build_shared_pool
    from src.experiments.shared_evaluation import evaluate, verify
    from src.experiments.defenses import conditions
    from src.tests.test_shared_pool import ToyEncoder
    pool, _ = pool_inputs
    shared = tmp_path/'shared_memory.json'
    build_shared_pool([pool], shared, kind='purchase_memory')
    feature = dict(asin=products[0]['id'], category='shampoo',
                   core_attributes=['gentle formula', 'moisturizing ingredients', 'easy rinsing'],
                   canonical_sentence='I want to buy shampoo with gentle formula and moisturizing ingredients.')
    manifest = tmp_path/'queries.json'
    save_json(manifest, build_feature_text_manifest(feature))
    frozen = dict(method='lucid', status='frozen', source=products[1], target=products[0],
        image=products[2]['image'], image_sha256=products[2]['image_sha256'], requested_steps=2,
        metadata=dict(method='lucid', status='success', output_image=products[2]['image'],
            output_image_sha256=products[2]['image_sha256'], attack=dict(steps_completed=2, steps_requested=2)))
    result = evaluate(frozen, tmp_path/'evaluation', pool_path=shared, query_manifest=manifest,
                      query_mode='text', encoder_factory=ToyEncoder, selected_conditions=[conditions()[0]])
    assert result['verification']['rankings_recomputed'] == 40
    plan = read_json(tmp_path/'evaluation/evaluation_plan.json')
    assert plan['query_mode'] == 'text' and plan['query_image_weight'] == 0
    rows = read_json(tmp_path/'evaluation/evaluation.json')
    assert all(row['retrieval_protocol']['query_mode'] == 'text' for row in rows)
    with pytest.raises(ValueError, match='复算指定的正式查询模式'):
        verify(tmp_path/'evaluation', encoder_factory=ToyEncoder, query_mode='joint')


def test_shared_feature_cache_keeps_actual_text_audit():
    from src.experiments.shared_batch import SharedFeatureCache
    class Encoder:
        snapshot = {'fixture': True}
        def encode_text(self, text):
            self.last_encoding_audit = dict(actual_text=text)
            return [1., 0.]
    cache = SharedFeatureCache(Encoder())
    cache.encode_text('first')
    cache.encode_text('second')
    cache.encode_text('first')
    assert cache.last_encoding_audit == dict(actual_text='first')


def test_matched_evaluator_requires_explicit_configuration_and_real_matching_weights(tmp_path, monkeypatch):
    from src.experiments import shared_evaluation as evaluation
    import src.visual._lucid.config as config_module
    import src.visual.encoders.clip as clip_module
    import src.visual.encoders.joint as joint_module
    weight = tmp_path/'fixture.bin'; weight.write_bytes(b'explicit-fixture-weights')
    model = SimpleNamespace(name='fixture-model', pretrained='fixture-weights', weight_path=str(weight))
    monkeypatch.setattr(config_module, 'resolve_optimization', lambda *a: SimpleNamespace(surrogates=[model]))
    monkeypatch.setattr(clip_module, 'MatchedClipEncoder', lambda *a: SimpleNamespace(snapshot={'fixture': True}))
    monkeypatch.setattr(joint_module, 'ChunkedClipEncoder', lambda base: SimpleNamespace(snapshot=copy.deepcopy(base.snapshot)))
    config = tmp_path/'evaluator.json'
    spec = dict(name=model.name, pretrained=model.pretrained, weight_path=weight.name)
    save_json(config, dict(setting='matched_public_encoder', surrogate=spec))
    encoder = evaluation.make_eval_encoder('cpu', config)
    protocol = encoder.snapshot['evaluation_protocol']
    assert protocol['matches_training_surrogate'] and protocol['black_box_transfer_claimed'] is False
    assert protocol['matching_training_surrogate_indices'] == [0]
    save_json(config, dict(setting='held_out_public_encoder', surrogate=spec))
    with pytest.raises(ValueError, match='不能作为黑盒迁移评估'):
        evaluation.make_eval_encoder('cpu', config)


def test_pipeline_cli_selects_text_manifest_and_only_no_defense(tmp_path, products, monkeypatch):
    from src.experiments import pipeline
    import src.experiments.shared_evaluation as evaluation
    root = tmp_path/'output/run'
    config = tmp_path/'pipeline.json'
    save_json(config, dict(schema_version=1, method='lucid', public_pair='fixture.json',
                           run_dir=str(root), evaluation={}))
    monkeypatch.setattr(pipeline, 'OUTPUT', tmp_path/'output')
    monkeypatch.setattr(pipeline, 'load_frozen', lambda *a: {'fixture': True})
    seen = {}
    monkeypatch.setattr(evaluation, 'evaluate', lambda *a, **kwargs: seen.update(kwargs))
    monkeypatch.setattr(evaluation, 'report', lambda *a: None)
    assert pipeline.main(['evaluate', '--config', str(config), '--query-mode', 'text',
                          '--query-manifest', str(tmp_path/'queries.json'), '--no-defense']) == 0
    assert seen['query_mode'] == 'text'
    assert seen['query_manifest'] == tmp_path/'queries.json'
    assert [c['kind'] for c in seen['selected_conditions']] == ['none']


def test_generation_text_mode_does_not_read_formal_manifest(products, tmp_path, monkeypatch):
    from pathlib import Path
    from src.experiments import pipeline
    source, target = products[1], dict(products[0], feature_text='A frozen public target feature.')
    selection = dict(source=source, target=target)
    root = tmp_path/'run'
    save_json(root/'selection/SELECTED_PAIR.json', selection)
    # 即使正式清单暂未生成，生成阶段也只读公开模式，不打开该评估私有输入。
    manifest = tmp_path/'not_generated_formal_manifest.json'
    config = dict(generation=dict(lucid=dict(steps=2)),
                  evaluation=dict(query_mode='text', query_manifest=str(manifest)))
    monkeypatch.setattr(pipeline, 'source_snapshot', lambda: {'fixture': True})
    def generate(source, target, output, **kwargs):
        return Path(source), dict(method='lucid', status='success', output_image=source,
            output_image_sha256=sha256_file(source), source_image_sha256=sha256_file(source),
            target_image_sha256=sha256_file(target), attack=dict(steps_completed=2, steps_requested=2))
    pipeline.generate(config, root, selection, 'lucid', device='cpu', generator=generate)
    request = read_json(root/'methods/lucid/QUERY_INPUT_REQUEST.json')
    assert request['query_image_required'] is False
    assert request['formal_instructions_read_during_generation'] is False
    assert not (root/'methods/lucid/QUERY_IMAGE_REQUEST.json').exists()


def test_joint_conversion_preserves_frozen_texts_ids_overlap_and_half_weights(features, products, tmp_path):
    from src.experiments.formal_queries import build_joint_manifest_from_text
    from src.experiments.query_contract import query_retriever_options
    text_manifest = build_feature_text_manifest(features)
    # 保留前导空格、大小写与换行，转换不重新生成或规范化任何正文。
    import hashlib
    import json
    text_manifest['instructions'][1] = '  ' + text_manifest['instructions'][1] + '\n'
    text_manifest['instructions_sha256'] = hashlib.sha256(json.dumps(
        text_manifest['instructions'], ensure_ascii=False, separators=(',', ':')).encode('utf-8')).hexdigest()
    original = copy.deepcopy(text_manifest)
    review = dict(no_explicit_product_name=True, no_brand=True, reviewer='实际核对的合成夹具')
    manifest = build_joint_manifest_from_text(text_manifest, products[2]['image'], review=review)
    assert text_manifest == original
    assert manifest['instructions'] == original['instructions']
    assert manifest['instructions_sha256'] == original['instructions_sha256']
    assert manifest['training_overlap'] == original['training_overlap']
    path = tmp_path/'joint_queries.json'; save_json(path, manifest)
    target = dict(products[0], id=features['asin'])
    queries = load_formal_queries(path, target, query_mode='joint')
    assert [query['task_id'] for query in queries] == [f'feature_{i:02d}' for i in range(1, 21)]
    assert [query['instruction'] for query in queries] == original['instructions']
    assert all(query['query_mode'] == 'joint' for query in queries)
    assert query_retriever_options(queries) == dict(query_mode='joint', query_image_weight=0.5)
    assert query_retriever_options(load_formal_queries(
        _save_text_manifest(tmp_path, original), target)) == dict(query_mode='text', query_image_weight=0.0)
    for query in queries:
        verify_query_input(query)


def _save_text_manifest(tmp_path, document):
    path = tmp_path/'text_queries.json'
    save_json(path, document)
    return path


def test_joint_cli_converts_existing_text_manifest_without_rephrasing(features, products, tmp_path):
    text = build_feature_text_manifest(features)
    original = _save_text_manifest(tmp_path, text)
    original_sha256 = sha256_file(original)
    review_file = tmp_path/'review.json'
    save_json(review_file, dict(no_explicit_product_name=True, no_brand=True, reviewer='合成夹具视觉核对'))
    output = tmp_path/'joint.json'
    args = ['--mode', 'feature-joint', '--text-manifest', str(original),
            '--image', products[2]['image'], '--image-review', str(review_file), '--output', str(output)]
    assert query_main(args) == 0 and query_main(args) == 0
    document = read_json(output)
    assert document['instructions'] == text['instructions'] and document['query_mode'] == 'joint'
    assert sha256_file(original) == original_sha256


@pytest.mark.parametrize('change', ['missing_image', 'wrong_image_digest', 'unreviewed', 'text_tampering'])
def test_joint_conversion_rejects_incomplete_or_changed_inputs(features, products, tmp_path, change):
    from src.experiments.formal_queries import build_joint_manifest_from_text
    manifest = build_feature_text_manifest(features)
    review = dict(no_explicit_product_name=True, no_brand=True, reviewer='合成核对')
    path, digest = products[2]['image'], products[2]['image_sha256']
    if change == 'missing_image':
        path = tmp_path/'not_generated.png'
    elif change == 'wrong_image_digest':
        digest = 'wrong'
    elif change == 'unreviewed':
        review = {}
    else:
        manifest['instructions'][0] += ' Unexpected extra words.'
    with pytest.raises((ValueError, FileNotFoundError)):
        build_joint_manifest_from_text(manifest, path, review=review, image_sha256=digest)


def test_joint_query_input_rejects_missing_or_changed_image(products, tmp_path):
    query = dict(query_mode='joint', instruction='A fixed feature request.')
    with pytest.raises(QueryImagePending):
        verify_query_input(query)
    query.update(query_image=products[2]['image'], query_image_sha256='wrong')
    with pytest.raises(ValueError, match='查询图片改变'):
        verify_query_input(query)


def test_joint_evaluation_and_fresh_verification_share_feature_texts_and_image(pool_inputs, products, features, tmp_path):
    from src.agent.memory.shared_pool import build_shared_pool
    from src.experiments.shared_evaluation import evaluate
    from src.experiments.defenses import conditions
    from src.experiments.formal_queries import build_joint_manifest_from_text
    from src.tests.test_shared_pool import ToyEncoder
    pool, _ = pool_inputs
    shared = tmp_path/'shared_memory.json'
    build_shared_pool([pool], shared, kind='purchase_memory')
    target = products[0]
    feature = dict(features, asin=target['id'])
    manifest = tmp_path/'queries.json'
    save_json(manifest, build_joint_manifest_from_text(build_feature_text_manifest(feature), products[2]['image'],
        review=dict(no_explicit_product_name=True, no_brand=True, reviewer='合成夹具')))
    frozen = dict(method='lucid', status='frozen', source=products[1], target=target,
        image=products[2]['image'], image_sha256=products[2]['image_sha256'], requested_steps=2,
        metadata=dict(method='lucid', status='success', output_image=products[2]['image'],
            output_image_sha256=products[2]['image_sha256'], attack=dict(steps_completed=2, steps_requested=2)))
    result = evaluate(frozen, tmp_path/'evaluation', pool_path=shared, query_manifest=manifest,
                      query_mode='joint', encoder_factory=ToyEncoder, selected_conditions=[conditions()[0]])
    assert result['verification']['rankings_recomputed'] == 40
    plan = read_json(tmp_path/'evaluation/evaluation_plan.json')
    assert plan['query_mode'] == 'joint' and plan['query_image_weight'] == 0.5
    rows = read_json(tmp_path/'evaluation/evaluation.json')
    assert all(row['retrieval_protocol']['query_mode'] == 'joint' for row in rows)
    assert [query['instruction'] for query in plan['queries']] == build_feature_text_manifest(feature)['instructions']


def test_policy_actual_multimodal_payload_contains_query_and_memory_image(products):
    from src.agent.shopping import llm
    requests, trace = [], {}
    def create(**kwargs):
        requests.append(kwargs)
        return dict(choices=[dict(message=dict(content='think[fixture]'), finish_reason='stop')])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    memory_asset = dict(Id='fixture_memory', Role='recalled_memory', LocalPath=products[1]['image'],
                        SHA256=products[1]['image_sha256'])
    assert llm('fixed feature request', 'fixture', client, 'fixture', query_image=products[2]['image'],
               memory_images=[memory_asset], trace=trace, max_retries=1) == 'think[fixture]'
    content = requests[0]['messages'][1]['content']
    assert len([item for item in content if item['type'] == 'image_url']) == 2
    assert [(image['Role'], image['SHA256']) for image in trace['InputImages']] == [
        ('query', products[2]['image_sha256']), ('recalled_memory', products[1]['image_sha256'])]
    assert trace['InputContent'][0]['text'] == 'fixed feature request'


def test_joint_shopping_cannot_silently_skip_retrieval_when_image_missing():
    from src.agent.shopping import webshop_run_rap
    class ForbiddenEnvironment:
        def step(self, *args):
            raise AssertionError('缺图时不能启动环境或调用模型')
    with pytest.raises(ValueError, match='缺少有效的正式参考图'):
        webshop_run_rap('fixture', '', '', '', ForbiddenEnvironment(), None, 'fixture', 'fixture',
            None, [], {}, 1, visual_retriever=SimpleNamespace(query_mode='joint'))
