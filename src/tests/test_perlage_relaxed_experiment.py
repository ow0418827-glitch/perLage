"""独立放宽实验的控制变量与共享库边界；不加载模型、不运行生成。"""
import copy

import pytest

from src.visual.image_utils import sha256_file
from src.visual.audit import save_json


def _bank_fixture(products):
    rows = []
    for index in range(189):
        product = products[index % len(products)]
        rows.append({
            'Id': f'entry_{index}', 'FinalASIN': product['id'],
            'RecordKind': 'simulated_gallery', 'Completed': False, 'Actions': [],
            'Instruction': '公开商品参考，未观察到购买',
            'Brand': '', 'Attributes': [f'真实特点 {index}'],
            'PrimaryImage': {'ASIN': product['id'], 'Title': product['title'],
                             'LocalPath': product['image'],
                             'SHA256': product['image_sha256']},
        })
    replacements = {}
    for name in ('baseline', 'relaxed', 'strong'):
        replacements[name] = {
            f'entry_{index}': {'image': products[2]['image'],
                             'sha256': products[2]['image_sha256']}
            for index in range(2)
        }
    return rows, replacements


def test_relaxed_import_keeps_existing_method_defaults_and_registry():
    from src.visual._perlage.latent import PerLAGELatentConfig
    from src.visual._perlage_no_pixel.pipeline import PerLAGENoPixelConfig
    from src.visual.optimization import METHODS
    before = PerLAGELatentConfig()
    import src.visual._perlage_no_pixel_relaxed  # noqa: F401
    assert PerLAGELatentConfig() == before
    assert (before.source_ssim_floor, before.source_mae_ceiling,
            before.latent_levels, before.feature_weight) == (.85, 12., 24, .02)
    assert PerLAGENoPixelConfig().pixel_optimization is False
    assert PerLAGENoPixelConfig().latent['auxiliary_levels'] == 0
    assert METHODS == ('lucid', 'PerLAGE', 'PerLAGE_no_pixel')


@pytest.mark.parametrize('changes', [
    {'auxiliary_levels': 1}, {'feature_weight': .02}, {'diffusion_weight': 3.},
    {'steps': 159}, {'source_ssim_floor': float('nan')},
    {'latent_learning_rate': float('inf')},
])
def test_relaxed_config_rejects_changed_experiment_semantics(changes):
    from src.visual._perlage_no_pixel_relaxed import RelaxedLatentConfig
    with pytest.raises(ValueError):
        RelaxedLatentConfig(**changes).validate()


def test_both_source_images_share_each_full_bank_without_rewriting_history(products, tmp_path):
    from src.experiments.perlage_relaxed_evaluation import build_banks
    entries, replacements = _bank_fixture(products)
    original = copy.deepcopy(entries)
    banks = build_banks(entries, tmp_path / 'pool.json', replacements)
    assert set(banks) == {'clean', 'baseline', 'relaxed', 'strong'}
    assert entries == original
    assert banks['clean'] == original
    for name in ('baseline', 'relaxed', 'strong'):
        assert len(banks[name]) == 189
        changed = [row['Id'] for row in banks[name] if 'RetrievalImage' in row]
        assert changed == ['entry_0', 'entry_1']
        assert [{k: v for k, v in row.items() if k != 'RetrievalImage'}
                for row in banks[name]] == banks['clean']
        assert banks[name][2:] == banks['clean'][2:]
        for row in banks[name][:2]:
            assert row['RetrievalImage']['SHA256'] == sha256_file(row['RetrievalImage']['LocalPath'])
            assert row['Actions'] == [] and row['Completed'] is False


def test_banks_reject_inconsistent_two_source_identity(products, tmp_path):
    from src.experiments.perlage_relaxed_evaluation import build_banks
    entries, replacements = _bank_fixture(products)
    replacements['strong']['entry_2'] = replacements['strong'].pop('entry_1')
    with pytest.raises(ValueError):
        build_banks(entries, tmp_path / 'pool.json', replacements)


def test_banks_reject_changed_frozen_image(products, tmp_path):
    from src.experiments.perlage_relaxed_evaluation import build_banks
    entries, replacements = _bank_fixture(products)
    replacements['strong']['entry_0']['sha256'] = '0' * 64
    with pytest.raises(ValueError):
        build_banks(entries, tmp_path / 'pool.json', replacements)


def test_banks_reject_small_group_pool(products, tmp_path):
    from src.experiments.perlage_relaxed_evaluation import build_banks
    entries, replacements = _bank_fixture(products)
    with pytest.raises(ValueError):
        build_banks(entries[:21], tmp_path / 'pool.json', replacements)


def test_incomplete_generation_stops_before_reading_pool_or_formal_queries(tmp_path, monkeypatch):
    import src.experiments.perlage_relaxed_evaluation as evaluation
    save_json(tmp_path / 'EXPERIMENT_PLAN.json', {
        'groups': list(evaluation.GROUPS), 'profiles': list(evaluation.PROFILES),
    })
    save_json(tmp_path / 'FROZEN_IMAGES.json', {
        'status': 'frozen', 'images': {'relaxed': {}},
    })
    read_paths = []
    original_read = evaluation.read_json

    def guarded_read(path):
        read_paths.append(str(path))
        if str(path).endswith(('shared_memory.json', 'queries.json')):
            raise AssertionError('不完整生成不允许访问正式评估输入')
        return original_read(path)

    monkeypatch.setattr(evaluation, 'read_json', guarded_read)
    with pytest.raises(ValueError):
        evaluation.evaluate(tmp_path)
    assert read_paths
    assert all(path.endswith(('EXPERIMENT_PLAN.json', 'FROZEN_IMAGES.json')) for path in read_paths)
