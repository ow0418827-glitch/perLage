"""Empirical Stress Tests for Milestone M2: PerLAGE_no_pixel Pipeline & Evidence Validation.

Author: Challenger 2 (challenger_m2_2)
Objective:
1. Verify 4 stages are executed sequentially without skipping.
2. Verify all embedded assertions (0 pixel steps, 0 residual, 0 auxiliary levels) are active and cannot be bypassed.
3. Test validate_complete against missing stages, hash mismatches, and tampered records.
4. Test integration with attack_completion.py.
"""
import copy
import hashlib
import json
import numpy as np
from PIL import Image
import pytest
from src.paths import ProjectPath as Path

from src.visual.audit import save_json, read_json
from src.visual.image_utils import sha256_file
from src.visual._perlage.components.evidence import STAGES
from src.visual.retrieval_objective import TRAINING_DATA_POLICY
from src.visual._perlage_no_pixel.evidence import (
    METHOD,
    DEFINITION,
    validate_complete,
    definition_for_method,
)
from src.visual._perlage_no_pixel.pipeline import (
    validate_definition,
    initialization_config,
    generate_perlage_no_pixel,
    PerLAGENoPixelConfig,
)
from src.visual.attack_completion import validate_attack_completion


# ==============================================================================
# Fixtures and Helpers
# ==============================================================================

@pytest.fixture
def valid_no_pixel_ready(products, tmp_path):
    """Constructs a fully valid ready.json dictionary for PerLAGE_no_pixel."""
    proof = tmp_path / 'proof.json'
    save_json(proof, {'public_binding': {'shadow_pool_used': False}})
    optimizer = tmp_path / 'optimizer.json'
    save_json(optimizer, {'config': {'pixel_clean_steps': 0, 'pixel_diffusion_steps': 0, 'pixel_levels': 0}})
    initialized = tmp_path / 'ready.json'
    save_json(initialized, {})

    stages = {name: dict(status='completed', files={str(proof): sha256_file(proof)}) for name in STAGES}
    stages['latent_generation_refinement'].update(steps_completed=40, steps_requested=40)

    final_image = products[1]['image']
    final_sha256 = products[1]['image_sha256']

    return dict(
        method=METHOD,
        definition=DEFINITION,
        status='completed',
        color_scheme='statistics',
        pixel_optimization=False,
        auxiliary_max_levels=0,
        latent_config={'auxiliary_levels': 0},
        optimization_mode='latent_only',
        shadow_pool_used=False,
        purchase_history_used=False,
        victim_memory_used=False,
        formal_queries_used=False,
        training_data_policy=TRAINING_DATA_POLICY,
        stages=stages,
        initialization_identity=str(proof),
        initialization_optimizer_identity=str(optimizer),
        initialization_evidence={str(p): sha256_file(p) for p in (proof, optimizer, initialized)},
        candidates={'final': dict(image=final_image, sha256=final_sha256)},
        source_sha256=products[1]['image_sha256'],
        target_sha256=products[0]['image_sha256'],
    )


@pytest.fixture
def dummy_pipeline_definition(products, tmp_path):
    """Constructs a valid definition dict for testing pipeline functions."""
    return dict(
        method=METHOD,
        definition=DEFINITION,
        color_scheme='statistics',
        training_data_policy=TRAINING_DATA_POLICY,
        pixel_optimization=False,
        data=dict(
            source=products[1]['image'],
            target=products[0]['image'],
            source_title=products[1]['title'],
            training_queries=['Find visually matching product.'],
        ),
        source_boxes=[[0, 0, 24, 24]],
        target_boxes=[[0, 0, 24, 24]],
        print_regions=[[2, 2, 8, 8]],
        initialization=dict(
            optimizer=dict(pixel_clean_steps=0, pixel_diffusion_steps=0, pixel_levels=0),
            generation=dict(steps=30),
            plan=dict(
                instruction='Refine packaging surface',
                editable_regions=[[0, 0, 1, 1]],
                protected_regions=[],
                reason='Test generation',
            ),
        ),
        latent=dict(steps=40, auxiliary_levels=0),
    )


# ==============================================================================
# 1. Pipeline Execution Structure (4 Stages Sequential Execution)
# ==============================================================================

def test_four_stages_executed_sequentially_without_skipping(products, tmp_path, dummy_pipeline_definition, monkeypatch):
    """Stress test: verify all 4 stages are called strictly in order 1 -> 2 -> 3 -> 4."""
    execution_order = []

    # Stage 1 Mock
    def mock_segment_owned_image(path, boxes, out_dir):
        execution_order.append(('stage_1_segment', Path(out_dir).name))
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        rec = out / 'record.json'
        fg = out / 'foreground.png'
        save_json(rec, {'boxes': boxes})
        Image.new('L', (24, 24), 255).save(fg)
        return np.ones((24, 24), dtype=bool), {'record': str(rec)}

    # Stage 2 Mock
    def mock_foreground_statistics(target_path, fg_mask):
        execution_order.append('stage_2_statistics')
        stats = {'rgb': [100, 150, 200], 'mean': [100, 150, 200]}
        body = np.ones((24, 24, 3), dtype=bool)
        return stats, body

    # Stage 3 Mocks
    def mock_prepare_generation_initialization(**kwargs):
        execution_order.append('stage_3_initialization')
        out = Path(kwargs['output'])
        out.mkdir(parents=True, exist_ok=True)
        (out / 'attack').mkdir(parents=True, exist_ok=True)

        run_id = out / 'run_identity.json'
        opt_id = out / 'attack/identity.json'
        meta_id = out / 'attack/metadata.json'
        ready_id = out / 'ready.json'

        save_json(run_id, {'public_binding': {'shadow_pool_used': False}})
        save_json(opt_id, {'config': {'pixel_clean_steps': 0, 'pixel_diffusion_steps': 0, 'pixel_levels': 0}})
        save_json(meta_id, {'pixel_residual_max_levels': 0, 'history': [{'phase': 'init_clean'}]})
        save_json(ready_id, {})

        # Anchor and final images are identical (0 residual)
        anchor_img = out / 'anchor.png'
        final_img = out / 'final.png'
        Image.new('RGB', (24, 24), (100, 150, 200)).save(anchor_img)
        Image.new('RGB', (24, 24), (100, 150, 200)).save(final_img)

        initial_dict = {
            'candidates': {
                'generation_only': {'image': str(anchor_img)},
                'natural_final': {'image': str(final_img)},
            }
        }
        return None, initial_dict

    def mock_fuse(data, initial, foreground, statistics, print_regions, root, config, base):
        execution_order.append('stage_3_fuse')
        fused_dir = root / 'color_harmonization'
        fused_dir.mkdir(parents=True, exist_ok=True)

        anchor = fused_dir / 'generative_anchor.png'
        init_img = fused_dir / 'initial.png'
        mask = fused_dir / 'edit_mask.png'
        meta = fused_dir / 'metadata.json'

        Image.new('RGB', (24, 24), (100, 150, 200)).save(anchor)
        Image.new('RGB', (24, 24), (100, 150, 200)).save(init_img)
        Image.new('L', (24, 24), 255).save(mask)
        save_json(meta, {'fused': True})
        return fused_dir, {}

    # Stage 4 Mock
    def mock_refine_harmonized_generation(**kwargs):
        execution_order.append('stage_4_latent_refine')
        out = Path(kwargs['output'])
        out.mkdir(parents=True, exist_ok=True)

        anchor = out / 'generative_anchor.png'
        adv = out / 'refined.png'
        meta_file = out / 'metadata.json'
        ident_file = out / 'identity.json'

        # Must match anchor RGB exactly (zero additive residual)
        Image.new('RGB', (24, 24), (100, 150, 200)).save(anchor)
        Image.new('RGB', (24, 24), (100, 150, 200)).save(adv)
        meta = {
            'status': 'completed',
            'auxiliary_max_levels': 0,
            'auxiliary_mae_levels': 0.0,
            'history': [{'step': 40}],
        }
        save_json(meta_file, meta)
        save_json(ident_file, {'steps': 40})
        return adv, meta

    # Mock encoder and purifier
    class DummySurrogate:
        device = 'cpu'
    class DummyEncoder:
        surrogate = DummySurrogate()
    class DummyPurifier:
        vae = type('VAE', (), {'enable_gradient_checkpointing': lambda self: None})()

    monkeypatch.setattr('src.visual._perlage_no_pixel.pipeline.segment_owned_image', mock_segment_owned_image)
    monkeypatch.setattr('src.visual._perlage_no_pixel.pipeline.foreground_statistics', mock_foreground_statistics)
    monkeypatch.setattr('src.visual._perlage_no_pixel.pipeline.prepare_generation_initialization', mock_prepare_generation_initialization)
    monkeypatch.setattr('src.visual._perlage_no_pixel.pipeline._fuse', mock_fuse)
    monkeypatch.setattr('src.visual._perlage_no_pixel.pipeline.refine_harmonized_generation', mock_refine_harmonized_generation)
    monkeypatch.setattr('src.visual.encoders.clip.make_encoder', lambda device: DummyEncoder())
    monkeypatch.setattr('src.visual.diffusion_purification.LatentDiffusionPurifier', lambda cfg, device: DummyPurifier())

    output_dir = tmp_path / 'no_pixel_run'
    final_image, ready = generate_perlage_no_pixel(
        source=dummy_pipeline_definition['data']['source'],
        target=dummy_pipeline_definition['data']['target'],
        output=output_dir,
        definition=dummy_pipeline_definition,
        device='cpu',
    )

    # Verify sequential ordering: Stage 1 (source, target) -> Stage 2 -> Stage 3 (init, fuse) -> Stage 4
    expected_order = [
        ('stage_1_segment', 'source'),
        ('stage_1_segment', 'target'),
        'stage_2_statistics',
        'stage_3_initialization',
        'stage_3_fuse',
        'stage_4_latent_refine',
    ]
    assert execution_order == expected_order

    # Verify returned ready artifact satisfies all properties
    assert ready['method'] == METHOD
    assert ready['definition'] == DEFINITION
    assert ready['status'] == 'completed'
    assert ready['pixel_optimization'] is False
    assert ready['auxiliary_max_levels'] == 0
    assert (output_dir / 'ready.json').is_file()
    assert (output_dir / 'method_identity.json').is_file()


def test_pipeline_interrupted_by_pause_request(products, tmp_path, dummy_pipeline_definition):
    """Verify that PAUSE_REQUESTED halts pipeline execution immediately when identity matches."""
    output_dir = tmp_path / 'paused_run'
    output_dir.mkdir(parents=True, exist_ok=True)
    identity_path = output_dir / 'method_identity.json'
    from src.visual._perlage_no_pixel.pipeline import PROJECT
    from src.paths import ProjectPath as Path
    implementations = [
        'src/visual/_perlage_no_pixel/pipeline.py',
        'src/visual/_perlage_no_pixel/evidence.py',
        'src/visual/_perlage/initialization.py',
        'src/visual/_perlage/latent.py',
        'src/visual/_perlage/components/optimizer.py',
        'src/visual/_perlage/components/core.py',
        'src/visual/_perlage/components/assets.py',
        'src/visual/_perlage/components/editor.py',
        'src/visual/_perlage/components/instruction.py',
        'src/visual/_perlage/components/mask.py',
        'src/visual/_perlage/components/colour.py',
        'src/visual/_perlage/components/segmentation.py',
        'src/visual/_perlage/components/losses.py',
        'src/visual/_perlage/components/latent_branch.py',
        'src/visual/_perlage/harmonization.py',
        'src/visual/encoders/backbone.py',
        'src/visual/encoders/clip.py',
        'src/visual/encoders/joint.py',
        'src/visual/retrieval_objective.py',
        'src/visual/robust_objectives.py',
        'src/visual/diffusion_components.py',
        'src/visual/diffusion_purification.py',
        'src/visual/config/optimization.json',
        'src/model_identity.py'
    ]
    config_digest = hashlib.sha256(json.dumps(dummy_pipeline_definition, sort_keys=True).encode('utf-8')).hexdigest()
    identity = dict(
        method=METHOD,
        definition=DEFINITION,
        configuration_sha256=config_digest,
        source_sha256=sha256_file(dummy_pipeline_definition['data']['source']),
        target_sha256=sha256_file(dummy_pipeline_definition['data']['target']),
        implementation={name: sha256_file(PROJECT / name) for name in implementations if (PROJECT / name).is_file()}
    )
    save_json(identity_path, identity)
    (output_dir / 'PAUSE_REQUESTED').touch()

    with pytest.raises(InterruptedError, match='实验已暂停'):
        generate_perlage_no_pixel(
            source=dummy_pipeline_definition['data']['source'],
            target=dummy_pipeline_definition['data']['target'],
            output=output_dir,
            definition=dummy_pipeline_definition,
            device='cpu',
        )



def test_pipeline_output_dir_safety(products, tmp_path, dummy_pipeline_definition):
    """Verify that non-empty output directory without method identity is rejected."""
    output_dir = tmp_path / 'dirty_dir'
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'stray_file.txt').write_text('dirty')

    with pytest.raises(ValueError, match='非空目录没有本次独立方法身份'):
        generate_perlage_no_pixel(
            source=dummy_pipeline_definition['data']['source'],
            target=dummy_pipeline_definition['data']['target'],
            output=output_dir,
            definition=dummy_pipeline_definition,
            device='cpu',
        )


# ==============================================================================
# 2. Embedded Assertions & Non-Bypassability
# ==============================================================================

@pytest.mark.parametrize('bad_param,bad_val', [
    ('pixel_optimization', True),
    ('pixel_optimization', 1),
])
def test_definition_rejects_pixel_optimization_flag(dummy_pipeline_definition, bad_param, bad_val):
    d = copy.deepcopy(dummy_pipeline_definition)
    d[bad_param] = bad_val
    with pytest.raises(ValueError, match='无像素方法禁止开启像素优化'):
        validate_definition(d)


@pytest.mark.parametrize('bad_key,bad_val', [
    ('pixel_clean_steps', 10),
    ('pixel_diffusion_steps', 20),
    ('pixel_levels', 2),
])
def test_definition_rejects_nonzero_optimizer_pixel_steps(dummy_pipeline_definition, bad_key, bad_val):
    d = copy.deepcopy(dummy_pipeline_definition)
    d['initialization']['optimizer'][bad_key] = bad_val
    with pytest.raises(ValueError, match='无像素模式不能配置像素优化'):
        validate_definition(d)


def test_definition_rejects_nonzero_latent_auxiliary_levels(dummy_pipeline_definition):
    d = copy.deepcopy(dummy_pipeline_definition)
    d['latent']['auxiliary_levels'] = 4
    with pytest.raises(ValueError, match='仅潜变量模式禁止像素优化：固定像素残差'):
        validate_definition(d)


def test_initialization_config_rejects_and_enforces_zeros(dummy_pipeline_definition):
    # Test valid definition zero values
    cfg = initialization_config(dummy_pipeline_definition)
    assert cfg.pixel_clean_steps == 0
    assert cfg.pixel_diffusion_steps == 0
    assert cfg.pixel_levels == 0

    # Test rejection on non-zero
    for key in ('pixel_clean_steps', 'pixel_diffusion_steps', 'pixel_levels'):
        bad = copy.deepcopy(dummy_pipeline_definition)
        bad['initialization']['optimizer'][key] = 5
        with pytest.raises(ValueError, match='仅潜变量模式不能配置像素优化'):
            initialization_config(bad)


def test_stage3_and_stage4_embedded_assertions(products, tmp_path, dummy_pipeline_definition, monkeypatch):
    """Stress test embedded runtime assertions in Stage 3 and Stage 4."""
    # Dummy mock for stage 1 & 2
    monkeypatch.setattr('src.visual._perlage_no_pixel.pipeline.segment_owned_image',
                        lambda p, b, o: (np.ones((24, 24), dtype=bool), {'record': 'r'}))
    monkeypatch.setattr('src.visual._perlage_no_pixel.pipeline.foreground_statistics',
                        lambda p, m: ({'rgb': [100, 100, 100], 'mean': [100, 100, 100]}, np.ones((24, 24, 3), dtype=bool)))
    class DummySurrogate:
        device = 'cpu'
    class DummyEncoder:
        surrogate = DummySurrogate()
    class DummyPurifier:
        vae = type('VAE', (), {'enable_gradient_checkpointing': lambda self: None})()
    monkeypatch.setattr('src.visual.encoders.clip.make_encoder', lambda device: DummyEncoder())
    monkeypatch.setattr('src.visual.diffusion_purification.LatentDiffusionPurifier', lambda cfg, device: DummyPurifier())

    # Case A: Stage 3 init metadata has non-zero pixel_residual_max_levels
    def bad_init_meta(**kwargs):
        out = Path(kwargs['output'])
        (out / 'attack').mkdir(parents=True, exist_ok=True)
        save_json(out / 'attack/metadata.json', {'pixel_residual_max_levels': 2, 'history': []})
        return None, {'candidates': {'generation_only': {'image': 'a'}, 'natural_final': {'image': 'b'}}}

    monkeypatch.setattr('src.visual._perlage_no_pixel.pipeline.prepare_generation_initialization', bad_init_meta)
    with pytest.raises(AssertionError, match='初始化残差必须为 0'):
        generate_perlage_no_pixel(
            source=dummy_pipeline_definition['data']['source'],
            target=dummy_pipeline_definition['data']['target'],
            output=tmp_path / 'bad_run_a',
            definition=dummy_pipeline_definition,
            device='cpu',
        )

    # Case B: Stage 3 init metadata history has pixel phase
    def bad_init_history(**kwargs):
        out = Path(kwargs['output'])
        (out / 'attack').mkdir(parents=True, exist_ok=True)
        save_json(out / 'attack/metadata.json', {'pixel_residual_max_levels': 0, 'history': [{'phase': 'pixel_clean'}]})
        return None, {'candidates': {'generation_only': {'image': 'a'}, 'natural_final': {'image': 'b'}}}

    monkeypatch.setattr('src.visual._perlage_no_pixel.pipeline.prepare_generation_initialization', bad_init_history)
    with pytest.raises(AssertionError, match='初始化优化器历史不得包含像素阶段'):
        generate_perlage_no_pixel(
            source=dummy_pipeline_definition['data']['source'],
            target=dummy_pipeline_definition['data']['target'],
            output=tmp_path / 'bad_run_b',
            definition=dummy_pipeline_definition,
            device='cpu',
        )


# ==============================================================================
# 3. Evidence Validation (validate_complete) Stress Tests
# ==============================================================================

def test_validate_complete_passes_on_valid_fixture(valid_no_pixel_ready):
    """Baseline check: valid fixture passes validate_complete cleanly."""
    res = validate_complete(valid_no_pixel_ready, valid_no_pixel_ready['candidates']['final']['image'])
    assert res['method'] == METHOD


@pytest.mark.parametrize('missing_stage', STAGES)
def test_validate_complete_rejects_missing_any_stage(valid_no_pixel_ready, missing_stage):
    r = copy.deepcopy(valid_no_pixel_ready)
    del r['stages'][missing_stage]
    with pytest.raises(ValueError, match='优化方法仍须完成四个自然优先阶段'):
        validate_complete(r)


def test_validate_complete_rejects_extra_unknown_stage(valid_no_pixel_ready):
    r = copy.deepcopy(valid_no_pixel_ready)
    r['stages']['pixel_post_processing'] = dict(status='completed', files={})
    with pytest.raises(ValueError, match='优化方法仍须完成四个自然优先阶段'):
        validate_complete(r)


def test_validate_complete_rejects_stage_not_completed(valid_no_pixel_ready):
    r = copy.deepcopy(valid_no_pixel_ready)
    r['stages']['foreground_segmentation']['status'] = 'running'
    with pytest.raises(ValueError, match='优化方法阶段没有完成：foreground_segmentation'):
        validate_complete(r)


def test_validate_complete_rejects_tampered_stage_file_hash(valid_no_pixel_ready, tmp_path):
    r = copy.deepcopy(valid_no_pixel_ready)
    # Tamper with file hash
    proof_path = r['initialization_identity']
    r['stages']['foreground_segmentation']['files'][proof_path] = '0000000000000000000000000000000000000000000000000000000000000000'
    with pytest.raises(ValueError, match='优化方法阶段证据缺失或改变：foreground_segmentation'):
        validate_complete(r)


def test_validate_complete_rejects_deleted_stage_file(valid_no_pixel_ready, tmp_path):
    r = copy.deepcopy(valid_no_pixel_ready)
    phantom = tmp_path / 'phantom.json'
    r['stages']['foreground_segmentation']['files'][str(phantom)] = 'abcd'
    with pytest.raises(ValueError, match='优化方法阶段证据缺失或改变：foreground_segmentation'):
        validate_complete(r)


def test_validate_complete_rejects_incomplete_latent_schedule(valid_no_pixel_ready):
    r = copy.deepcopy(valid_no_pixel_ready)
    r['stages']['latent_generation_refinement']['steps_completed'] = 39
    r['stages']['latent_generation_refinement']['steps_requested'] = 40
    with pytest.raises(ValueError, match='优化方法生成潜变量细化日程未完成'):
        validate_complete(r)


@pytest.mark.parametrize('bad_zero_pixel_key,bad_zero_pixel_val,err_match', [
    ('pixel_optimization', True, '无像素方法不能包含像素优化'),
    ('auxiliary_max_levels', 1, '无像素方法不能包含固定像素残差'),
    ('auxiliary_max_levels', 4, '无像素方法不能包含固定像素残差'),
    ('optimization_mode', 'full_pixel', '无像素方法优化模式必须为 latent_only'),
])
def test_validate_complete_rejects_nonzero_pixel_residual_flags(valid_no_pixel_ready, bad_zero_pixel_key, bad_zero_pixel_val, err_match):
    r = copy.deepcopy(valid_no_pixel_ready)
    r[bad_zero_pixel_key] = bad_zero_pixel_val
    with pytest.raises(ValueError, match=err_match):
        validate_complete(r)


def test_validate_complete_rejects_latent_config_auxiliary_levels(valid_no_pixel_ready):
    r = copy.deepcopy(valid_no_pixel_ready)
    r['latent_config'] = {'auxiliary_levels': 2}
    with pytest.raises(ValueError, match='无像素方法潜变量配置不能包含辅助像素预算'):
        validate_complete(r)


@pytest.mark.parametrize('bad_opt_key,bad_opt_val', [
    ('pixel_clean_steps', 20),
    ('pixel_diffusion_steps', 60),
    ('pixel_levels', 4),
])
def test_validate_complete_rejects_optimizer_identity_pixel_steps(valid_no_pixel_ready, tmp_path, bad_opt_key, bad_opt_val):
    r = copy.deepcopy(valid_no_pixel_ready)
    bad_opt = tmp_path / f'bad_opt_{bad_opt_key}.json'
    save_json(bad_opt, {'config': {bad_opt_key: bad_opt_val}})
    r['initialization_optimizer_identity'] = str(bad_opt)
    r['initialization_evidence'][str(bad_opt)] = sha256_file(bad_opt)
    with pytest.raises(ValueError, match='无像素方法不能包含像素优化步数或等级'):
        validate_complete(r)


def test_validate_complete_rejects_tampered_initialization_evidence(valid_no_pixel_ready):
    r = copy.deepcopy(valid_no_pixel_ready)
    # Alter sha256 hash in initialization_evidence dict
    proof_path = r['initialization_identity']
    r['initialization_evidence'][proof_path] = 'tampered_hash_00000000000000000000000000'
    with pytest.raises(ValueError, match='优化方法初始化证据改变'):
        validate_complete(r)


def test_validate_complete_rejects_initialization_with_shadow_pool(valid_no_pixel_ready, tmp_path):
    r = copy.deepcopy(valid_no_pixel_ready)
    tainted_proof = tmp_path / 'tainted_proof.json'
    save_json(tainted_proof, {'public_binding': {'shadow_pool_used': True}})
    r['initialization_identity'] = str(tainted_proof)
    r['initialization_evidence'][str(tainted_proof)] = sha256_file(tainted_proof)
    with pytest.raises(ValueError, match='优化方法不能复用含竞争池的初始化'):
        validate_complete(r)


def test_validate_complete_rejects_final_image_mismatches(valid_no_pixel_ready, tmp_path):
    # Case A: final image path does not exist
    r = copy.deepcopy(valid_no_pixel_ready)
    r['candidates']['final']['image'] = str(tmp_path / 'non_existent.png')
    with pytest.raises(ValueError, match='优化方法最终图片不符'):
        validate_complete(r)

    # Case B: final image sha256 mismatch
    r = copy.deepcopy(valid_no_pixel_ready)
    r['candidates']['final']['sha256'] = 'wrong_hash'
    with pytest.raises(ValueError, match='优化方法最终图片不符'):
        validate_complete(r)

    # Case C: passed image parameter does not match candidates.final
    r = copy.deepcopy(valid_no_pixel_ready)
    other_img = tmp_path / 'other.png'
    Image.new('RGB', (24, 24), (0, 0, 0)).save(other_img)
    with pytest.raises(ValueError, match='当前图片没有绑定优化方法完整完成记录'):
        validate_complete(r, image=other_img)


def test_validate_complete_rejects_wrong_method_identity(valid_no_pixel_ready):
    r = copy.deepcopy(valid_no_pixel_ready)
    r['method'] = 'PerLAGE'
    with pytest.raises(ValueError, match='方法身份不符合'):
        validate_complete(r, method=METHOD)

    r = copy.deepcopy(valid_no_pixel_ready)
    r['method'] = 'lucid'
    with pytest.raises(ValueError, match='方法身份不符合'):
        validate_complete(r)


def test_validate_complete_rejects_wrong_definition(valid_no_pixel_ready):
    r = copy.deepcopy(valid_no_pixel_ready)
    r['definition'] = 'perlage_foreground_statistics_harmonization_perceptual_latent_v1'
    with pytest.raises(ValueError, match='完成记录中的方法定义不符合'):
        validate_complete(r)


def test_validate_complete_rejects_non_dictionary():
    with pytest.raises(ValueError, match='完成记录必须为字典对象'):
        validate_complete(None)
    with pytest.raises(ValueError, match='完成记录必须为字典对象'):
        validate_complete("invalid_string")


def test_validate_complete_rejects_violation_of_training_boundaries(valid_no_pixel_ready):
    for boundary in ('shadow_pool_used', 'purchase_history_used', 'victim_memory_used', 'formal_queries_used'):
        r = copy.deepcopy(valid_no_pixel_ready)
        r[boundary] = True
        with pytest.raises(ValueError, match='无像素感知方法缺少独立训练边界：' + boundary):
            validate_complete(r)


# ==============================================================================
# 4. Integration with attack_completion.py Stress Tests
# ==============================================================================

def test_attack_completion_integration_with_no_pixel(valid_no_pixel_ready, products):
    """Test full integration with validate_attack_completion for PerLAGE_no_pixel."""
    final_img = valid_no_pixel_ready['candidates']['final']['image']
    final_sha = valid_no_pixel_ready['candidates']['final']['sha256']

    metadata = dict(
        method='PerLAGE_no_pixel',
        definition=DEFINITION,
        status='success',
        output_image=final_img,
        output_image_sha256=final_sha,
        source_image_sha256=products[1]['image_sha256'],
        target_image_sha256=products[0]['image_sha256'],
        shadow_pool_used=False,
        perlage_ready=valid_no_pixel_ready,
    )

    # Valid run passes cleanly
    validate_attack_completion(metadata, requested_steps=40)

    # Normalized lowercase method name also works
    metadata_lower = copy.deepcopy(metadata)
    metadata_lower['method'] = 'perlage_no_pixel'
    validate_attack_completion(metadata_lower, requested_steps=40)


def test_attack_completion_rejects_tampered_metadata(valid_no_pixel_ready, products, tmp_path):
    final_img = valid_no_pixel_ready['candidates']['final']['image']
    final_sha = valid_no_pixel_ready['candidates']['final']['sha256']

    base_metadata = dict(
        method='PerLAGE_no_pixel',
        definition=DEFINITION,
        status='success',
        output_image=final_img,
        output_image_sha256=final_sha,
        source_image_sha256=products[1]['image_sha256'],
        target_image_sha256=products[0]['image_sha256'],
        shadow_pool_used=False,
        perlage_ready=valid_no_pixel_ready,
    )

    # 1. status != 'success'
    m = copy.deepcopy(base_metadata)
    m['status'] = 'failed'
    with pytest.raises(ValueError, match='图片方法尚未正常完成'):
        validate_attack_completion(m, 40)

    # 2. output_image missing or tampered sha256
    m = copy.deepcopy(base_metadata)
    m['output_image_sha256'] = 'tampered'
    with pytest.raises(ValueError, match='最终图片缺失或摘要改变'):
        validate_attack_completion(m, 40)

    # 3. shadow pool used
    m = copy.deepcopy(base_metadata)
    m['shadow_pool_used'] = True
    with pytest.raises(ValueError, match='完整方法不得读入影子池或私有共享记忆池'):
        validate_attack_completion(m, 40)

    # 4. Source image sha256 mismatch with perlage_ready
    m = copy.deepcopy(base_metadata)
    m['source_image_sha256'] = 'mismatched_source_sha256'
    with pytest.raises(ValueError, match='四阶段证据与本次公开图对不符'):
        validate_attack_completion(m, 40)

    # 5. Target image sha256 mismatch with perlage_ready
    m = copy.deepcopy(base_metadata)
    m['target_image_sha256'] = 'mismatched_target_sha256'
    with pytest.raises(ValueError, match='四阶段证据与本次公开图对不符'):
        validate_attack_completion(m, 40)

    # 6. Expected input values mismatch
    m = copy.deepcopy(base_metadata)
    with pytest.raises(ValueError, match='完成记录未绑定本次冻结输入'):
        validate_attack_completion(m, 40, expected={'model_frozen_id': 'exp_123'})

    # 7. Tampered ready record embedded inside metadata
    m = copy.deepcopy(base_metadata)
    m['perlage_ready']['pixel_optimization'] = True
    with pytest.raises(ValueError, match='无像素方法不能包含像素优化'):
        validate_attack_completion(m, 40)
