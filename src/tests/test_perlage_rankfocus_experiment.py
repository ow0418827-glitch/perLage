"""排名优先试验的目标、冻结接口和完整共享库边界；不加载实际模型。"""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.visual.audit import save_json
from src.visual.image_utils import sha256_file


def _public_record():
    return dict(id='public_source', title='Public fixture product', brand='Known brand',
                attributes=['first real feature', 'second real feature'],
                feature_text='A real public product description.',
                provenance='synthetic_public_test_fixture', feature_evidence=[])


def test_public_full_key_equals_retrieval_text_without_history():
    from src.visual._perlage_no_pixel_rankfocus import build_source_public_key
    from src.visual.visual_retrieval import memory_index_text
    record = _public_record()
    entry = dict(FinalASIN=record['id'], Brand=record['brand'],
                 Attributes=record['attributes'], FeatureText=record['feature_text'],
                 Actions=['this content must never enter training'],
                 PrimaryImage=dict(ASIN=record['id'], Title=record['title']))
    assert build_source_public_key(record, record['title']) == memory_index_text(entry, 'product_feature')
    with pytest.raises(ValueError):
        build_source_public_key(dict(record, Actions=entry['Actions']), record['title'])


def test_compensated_teacher_reconstructs_joint_query_and_preserves_gradient(monkeypatch):
    import src.visual._perlage_no_pixel_rankfocus.pipeline as pipeline
    own = F.normalize(torch.tensor([[.2, .6, .7]]), dim=-1)
    text = F.normalize(torch.tensor([[.7, .5, .1]]), dim=-1)

    class SyntheticTextEncoder:
        def __init__(self, encoder):
            self.last_encoding_audit = {'synthetic_fixture': True}

        def encode_text(self, value):
            return (own if value == 'public complete key' else text).numpy().ravel()

    monkeypatch.setattr(pipeline, 'ChunkedClipEncoder', SyntheticTextEncoder)
    surrogate = SimpleNamespace(device='cpu', snapshot=lambda: {'synthetic_fixture': True})
    objective = pipeline.PublicFullKeyObjective(
        surrogate, torch.tensor([[1., .2, .1]]), 'public complete key', ['fixed training features'])
    reconstructed = F.normalize(objective.compensated_features + own, dim=-1)
    assert torch.allclose(reconstructed, objective.query_features, atol=1e-6)
    assert torch.allclose(objective.scores(objective.compensated_features), torch.ones(1, 1), atol=1e-6)
    features = torch.tensor([[.1, .4, .8]], requires_grad=True)
    loss = -.5 * objective.scores(features).mean() - .5 * objective.compensated_scores(features).mean()
    loss.backward()
    assert torch.isfinite(features.grad).all() and features.grad.norm() > 0


def test_new_import_keeps_old_methods_and_relaxed_defaults():
    from src.visual._perlage.latent import PerLAGELatentConfig
    from src.visual._perlage_no_pixel_relaxed import RELAXATION_PRESETS
    from src.visual.optimization import METHODS
    old = PerLAGELatentConfig()
    relaxed = copy.deepcopy(RELAXATION_PRESETS)
    import src.visual._perlage_no_pixel_rankfocus  # noqa: F401
    assert PerLAGELatentConfig() == old
    assert old.source_ssim_floor == .85 and old.latent_levels == 24
    assert RELAXATION_PRESETS == relaxed
    assert relaxed['strong']['steps'] == 240
    assert METHODS == ('lucid', 'PerLAGE', 'PerLAGE_no_pixel')


def test_both_predefined_profiles_complete_thousand_steps_without_pixels():
    from src.visual._perlage_no_pixel_rankfocus import RANKFOCUS_PRESETS, RankFocusLatentConfig
    for values in RANKFOCUS_PRESETS.values():
        RankFocusLatentConfig(**values).validate()
        assert values['steps'] == 1000
        assert values['latent_working_size'] == 512
        assert values['auxiliary_levels'] == values['feature_weight'] == values['diffusion_weight'] == 0
    assert RANKFOCUS_PRESETS['rankfocus']['source_ssim_floor'] > RANKFOCUS_PRESETS['rankfocus_wide']['source_ssim_floor']


@pytest.mark.parametrize('changes', [
    {'auxiliary_levels': 1}, {'feature_weight': .02}, {'diffusion_weight': 3.},
    {'latent_learning_rate': float('nan')}, {'source_mae_ceiling': float('inf')},
])
def test_rankfocus_rejects_pixel_and_nonfinite_changes(changes):
    from src.visual._perlage_no_pixel_rankfocus import RankFocusLatentConfig
    with pytest.raises(ValueError):
        RankFocusLatentConfig(**changes).validate()


def _synthetic_banks(products):
    from src.experiments.perlage_rankfocus_evaluation import BANKS
    rows = []
    for index in range(189):
        p = products[index % len(products)]
        rows.append(dict(Id=f'entry_{index}', FinalASIN=p['id'], RecordKind='simulated_gallery',
                         Completed=False, Actions=[], Brand='', Attributes=[f'public attribute {index}'],
                         PrimaryImage=dict(ASIN=p['id'], Title=p['title'], LocalPath=p['image'], SHA256=p['image_sha256'])))
    replacements = {name: {f'entry_{i}': dict(image=products[2]['image'], sha256=products[2]['image_sha256'])
                            for i in range(2)} for name in BANKS if name != 'clean'}
    return rows, replacements


def test_six_banks_replace_same_two_images_and_preserve_other_187(products, tmp_path):
    from src.experiments.perlage_rankfocus_evaluation import build_banks, BANKS
    entries, replacements = _synthetic_banks(products)
    original = copy.deepcopy(entries)
    banks = build_banks(entries, tmp_path / 'pool.json', replacements)
    assert entries == original and set(banks) == set(BANKS) and len(banks) == 6
    for name in BANKS[1:]:
        assert len(banks[name]) == 189
        assert [r['Id'] for r in banks[name] if 'RetrievalImage' in r] == ['entry_0', 'entry_1']
        assert banks[name][2:] == banks['clean'][2:]
        assert [{k: v for k, v in row.items() if k != 'RetrievalImage'} for row in banks[name]] == banks['clean']
    replacements['lucid']['entry_2'] = replacements['lucid'].pop('entry_1')
    with pytest.raises(ValueError):
        build_banks(entries, tmp_path / 'pool.json', replacements)


def test_partial_generation_blocks_formal_input_access(tmp_path, monkeypatch):
    import src.experiments.perlage_rankfocus_evaluation as evaluation
    save_json(tmp_path / 'EXPERIMENT_PLAN.json', dict(groups=list(evaluation.GROUPS), profiles=list(evaluation.PROFILES)))
    save_json(tmp_path / 'FROZEN_IMAGES.json', dict(status='frozen', images={'rankfocus': {}}))
    original_read, read_paths = evaluation.read_json, []

    def checked_read(path):
        read_paths.append(str(path))
        if str(path).endswith(('shared_memory.json', 'queries.json')):
            raise AssertionError('Incomplete images accessed formal evaluation data')
        return original_read(path)

    monkeypatch.setattr(evaluation, 'read_json', checked_read)
    with pytest.raises(ValueError):
        evaluation.evaluate(tmp_path)
    assert all(path.endswith(('EXPERIMENT_PLAN.json', 'FROZEN_IMAGES.json')) for path in read_paths)


@pytest.mark.parametrize('stage', ['evaluate', 'verify'])
def test_completed_evaluation_resumes_without_new_encoder_or_pid_rewrite(tmp_path, monkeypatch, stage):
    import src.experiments.perlage_rankfocus_evaluation as evaluation
    import src.experiments.shared_evaluation as shared_evaluation
    save_json(tmp_path / 'EVALUATION_INPUTS.json', {})
    marker = tmp_path / ('RETRIEVAL_COMPLETED.json' if stage == 'evaluate' else 'INDEPENDENT_VERIFICATION.json')
    save_json(marker, dict(pid=100, status='completed' if stage == 'evaluate' else 'passed'))
    original = marker.read_bytes()
    checked = []
    monkeypatch.setattr(evaluation, '_inputs', lambda run: {'synthetic_fixture': True})
    monkeypatch.setattr(evaluation, '_validate_retrieval', lambda *args: checked.append('retrieval') or {'pid': 100})
    monkeypatch.setattr(evaluation, '_validate_verification', lambda *args: checked.append('verification') or {'pid': 101})

    def forbidden_encoder(*args, **kwargs):
        raise AssertionError('Completed evaluation must not rerun encoding')

    monkeypatch.setattr(shared_evaluation, 'make_eval_encoder', forbidden_encoder)
    getattr(evaluation, stage)(tmp_path)
    assert marker.read_bytes() == original
    assert 'retrieval' in checked
    if stage == 'verify':
        assert 'verification' in checked


def test_latent_loop_accepts_soft_halo_but_preserves_print_and_background(tmp_path, monkeypatch):
    """用可微小模型跑实际更新循环，验证光晕超过旧阈值时仍接受合法更新。"""
    from PIL import Image
    import src.visual._perlage_no_pixel_rankfocus.pipeline as pipeline
    import src.visual.encoders.clip as clip_module
    source = tmp_path / 'source.png'
    target = tmp_path / 'target.png'
    pixels = np.full((24, 24, 3), 128, dtype=np.uint8)
    Image.fromarray(pixels).save(source)
    Image.fromarray(np.full_like(pixels, (200, 80, 80))).save(target)
    foreground = np.zeros((24, 24), dtype=bool)
    foreground[3:21, 3:21] = True
    prefix = tmp_path / 'prefix'
    for subdir in ('color_harmonization', 'foreground_segmentation/source'):
        (prefix / subdir).mkdir(parents=True)
    Image.fromarray(pixels).save(prefix / 'color_harmonization/generative_anchor.png')
    Image.fromarray(np.full((24, 24), 255, np.uint8)).save(prefix / 'color_harmonization/edit_mask.png')
    Image.fromarray(np.repeat((foreground * 255).astype(np.uint8)[..., None], 3, axis=-1)).save(
        prefix / 'foreground_segmentation/source/foreground.png')
    record = _public_record()
    definition = dict(data=dict(source=str(source), target=str(target), source_title=record['title'],
                                training_queries=['fixed original features']), print_regions=[[.35, .35, .65, .65]])
    original_ready = dict(source_title_sha256='synthetic')
    prefix_binding = dict(stages={name: dict(status='completed', files={'synthetic': 'fixture'})
                                  for name in pipeline.STAGES[:3]})
    monkeypatch.setattr(pipeline, 'PROJECT', tmp_path)
    monkeypatch.setattr(pipeline, 'validate_prefix', lambda *args: (
        prefix, original_ready, definition['data'], source, source, prefix_binding))
    monkeypatch.setattr(pipeline, '_files', lambda paths: {str(p): 'synthetic' for p in paths})
    monkeypatch.setattr(pipeline, '_VaeOnly', lambda device: SimpleNamespace(identity={'synthetic': True}))
    monkeypatch.setattr(pipeline, 'validate_complete', lambda ready, image=None: ready)
    config = copy.deepcopy(pipeline.RANKFOCUS_PRESETS['rankfocus_wide'])
    config.update(steps=10, check_every=10)
    monkeypatch.setitem(pipeline.RANKFOCUS_PRESETS, 'rankfocus_wide', config)

    class Surrogate:
        device = 'cpu'

        def snapshot(self):
            return {'synthetic': True}

        def encode_image(self, image):
            return SimpleNamespace(global_embedding=image.mean(dim=(-1, -2)))

    monkeypatch.setattr(clip_module, 'make_encoder', lambda **kwargs: SimpleNamespace(surrogate=Surrogate()))

    class SyntheticLatentBranch:
        def __init__(self, anchor, mask, vae, config):
            self.mask, self.config = mask, config
            self.delta = torch.zeros_like(anchor, requires_grad=True)

        def residual(self):
            return self.mask * self.config.latent_levels / 255 * torch.tanh(self.delta)

    class SyntheticObjective:
        def __init__(self, *args):
            self.audit = {'synthetic': True}

        def scores(self, features):
            return features[:, :1]

        compensated_scores = scores

        def diagnostics(self, features):
            return {'synthetic': True}

    monkeypatch.setattr(pipeline, 'SourceLatentBranch', SyntheticLatentBranch)
    monkeypatch.setattr(pipeline, 'PublicFullKeyObjective', SyntheticObjective)
    halo_mask = np.zeros((24, 24), bool)
    halo_mask[4, 4] = True
    indices = np.indices(halo_mask.shape)
    monkeypatch.setattr(pipeline, 'prepare_print_context', lambda *args: dict(halo=halo_mask, nearest=indices))
    monkeypatch.setattr(pipeline, 'halo_error', lambda context, image: 0. if np.array_equal(image, pixels) else 999.)
    image, ready = pipeline.generate_rankfocus_no_pixel(source, target, tmp_path / 'run', definition=definition,
        prefix_root=prefix, source_public_record=record, profile='rankfocus_wide', device='cpu')
    metadata = pipeline._read(tmp_path / 'run/metadata.json')
    result = np.asarray(Image.open(image).convert('RGB'))
    print_mask = pipeline.print_region_mask(pixels.shape, definition['print_regions'])
    assert metadata['accepted_updates'] > 0
    assert metadata['best']['natural_halo_gate'] is False
    assert metadata['best']['quality_ok'] is True
    assert metadata['best']['selection'][0] == 1
    assert np.array_equal(result[print_mask], pixels[print_mask])
    assert np.array_equal(result[~foreground], pixels[~foreground])
    assert np.any(result[foreground & ~print_mask] != pixels[foreground & ~print_mask])
    assert ready['auxiliary_max_levels'] == 0


def _load_queue():
    location = Path(__file__).resolve().parents[2] / 'tools/run_perlage_rankfocus_groups67_20261111.py'
    spec = importlib.util.spec_from_file_location('rankfocus_queue_test', location)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('fault', ['wrong_group', 'wrong_profile', 'unfinished_schedule'])
def test_queue_rejects_cross_group_profile_or_short_schedule(products, tmp_path, monkeypatch, fault):
    import src.visual._perlage_no_pixel_rankfocus as package
    queue = _load_queue()
    monkeypatch.setattr(package, 'validate_rankfocus_complete', lambda ready, image: ready)
    config = copy.deepcopy(package.RANKFOCUS_PRESETS['rankfocus'])
    definition = dict(data=dict(source=products[0]['image'], target=products[1]['image']))
    record = _public_record()
    root = queue.image_root(tmp_path, 'rankfocus', '06_jeans')
    root.mkdir(parents=True)
    (root / 'adversarial.png').write_bytes(Path(products[0]['image']).read_bytes())
    identity = root / 'identity.json'
    save_json(identity, dict(public_definition=definition, source_public_record=record))
    ready = dict(profile='rankfocus', latent_config=config,
                 stages={'latent_generation_refinement': dict(steps_requested=1000)},
                 source_sha256=products[0]['image_sha256'], target_sha256=products[1]['image_sha256'],
                 method_identity=str(identity))
    if fault == 'wrong_group':
        ready['source_sha256'] = products[2]['image_sha256']
    elif fault == 'wrong_profile':
        ready['profile'] = 'rankfocus_wide'
    else:
        ready['stages']['latent_generation_refinement']['steps_requested'] = 990
    save_json(root / 'ready.json', ready)
    plan = dict(profile_configs={'rankfocus': config}, generation_inputs={'06_jeans': dict(definition=definition, source_public_record=record)})
    with pytest.raises(ValueError):
        queue.validate_image(tmp_path, 'rankfocus', '06_jeans', plan)
