"""单组分布式潜变量实验的区域预算、早停及完整共享池边界。"""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from src.visual.audit import save_json


def test_new_method_preserves_existing_methods_and_frozen_presets():
    from src.visual._perlage.latent import PerLAGELatentConfig
    from src.visual._perlage_no_pixel_rankfocus import RANKFOCUS_PRESETS
    from src.visual._perlage_no_pixel_relaxed import RELAXATION_PRESETS
    from src.visual.optimization import METHODS
    defaults = PerLAGELatentConfig()
    frozen_rank = copy.deepcopy(RANKFOCUS_PRESETS)
    frozen_relaxed = copy.deepcopy(RELAXATION_PRESETS)
    import src.visual._perlage_no_pixel_distributed  # noqa: F401
    assert PerLAGELatentConfig() == defaults
    assert defaults.source_ssim_floor == .85 and defaults.latent_levels == 24
    assert RANKFOCUS_PRESETS == frozen_rank
    assert RELAXATION_PRESETS == frozen_relaxed
    assert METHODS == ('lucid', 'PerLAGE', 'PerLAGE_no_pixel')


def _synthetic_banks(products):
    from src.experiments.perlage_distributed_evaluation import BANKS
    entries = []
    for index in range(189):
        product = products[index % len(products)]
        entries.append(dict(Id=f'entry_{index}', FinalASIN=product['id'],
            RecordKind='simulated_gallery', Completed=False, Actions=[],
            Brand='', Attributes=[f'public feature {index}'],
            PrimaryImage=dict(ASIN=product['id'], Title=product['title'],
                LocalPath=product['image'], SHA256=product['image_sha256'])))
    asset = dict(image=products[2]['image'], sha256=products[2]['image_sha256'])
    replacements = {name: {'entry_0': copy.deepcopy(asset)} for name in BANKS if name != 'clean'}
    return entries, replacements


def test_seven_banks_replace_only_one_source_and_preserve_other_188(products, tmp_path):
    from src.experiments.perlage_distributed_evaluation import BANKS, GROUPS, build_banks
    assert GROUPS == ('07_shampoo',)
    assert len(BANKS) == 7
    entries, replacements = _synthetic_banks(products)
    original = copy.deepcopy(entries)
    banks = build_banks(entries, tmp_path / 'pool.json', replacements)
    assert entries == original and set(banks) == set(BANKS)
    for name in BANKS[1:]:
        bank = banks[name]
        assert len(bank) == 189 and len({row['Id'] for row in bank}) == 189
        assert [row['Id'] for row in bank if 'RetrievalImage' in row] == ['entry_0']
        assert bank[1:] == banks['clean'][1:]
        assert [{k: v for k, v in row.items() if k != 'RetrievalImage'} for row in bank] == banks['clean']


@pytest.mark.parametrize('fault', ['second_source', 'different_source', 'small_pool'])
def test_banks_reject_scope_expansion_and_group_subpool(products, tmp_path, fault):
    from src.experiments.perlage_distributed_evaluation import build_banks
    entries, replacements = _synthetic_banks(products)
    if fault == 'second_source':
        for changes in replacements.values():
            changes['entry_1'] = copy.deepcopy(changes['entry_0'])
    elif fault == 'different_source':
        replacements['distributed_balanced']['entry_1'] = replacements['distributed_balanced'].pop('entry_0')
    else:
        entries = entries[:21]
    with pytest.raises(ValueError):
        build_banks(entries, tmp_path / 'pool.json', replacements)


def test_partial_three_image_freeze_cannot_read_formal_queries_or_pool(tmp_path, monkeypatch):
    import src.experiments.perlage_distributed_evaluation as evaluation
    save_json(tmp_path / 'EXPERIMENT_PLAN.json', dict(groups=['07_shampoo'], profiles=list(evaluation.PROFILES)))
    save_json(tmp_path / 'FROZEN_IMAGES.json', dict(status='frozen', images={'foreground_control': {}}))
    original_read = evaluation.read_json
    reads = []

    def checked_read(path):
        reads.append(str(path))
        if str(path).endswith(('shared_memory.json', 'queries.json')):
            raise AssertionError('未全部成图前不能访问正式评估输入')
        return original_read(path)

    monkeypatch.setattr(evaluation, 'read_json', checked_read)
    with pytest.raises(ValueError):
        evaluation.evaluate(tmp_path)
    assert all(path.endswith(('EXPERIMENT_PLAN.json', 'FROZEN_IMAGES.json')) for path in reads)


def _load_queue():
    path = Path(__file__).resolve().parents[2] / 'tools/run_perlage_distributed_group7_20261111.py'
    spec = importlib.util.spec_from_file_location('distributed_queue_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_queue_is_hard_limited_to_the_authorized_single_group():
    queue = _load_queue()
    assert queue.GROUPS == ('07_shampoo',)
    assert 'group7' in queue.DEFAULT_RUN.name


def test_queue_rejects_an_experiment_plan_that_adds_the_other_group(tmp_path):
    queue = _load_queue()
    save_json(tmp_path / 'EXPERIMENT_PLAN.json', dict(groups=['06_jeans', '07_shampoo'],
        profiles=list(queue.PROFILES), shopping_enabled=False))
    with pytest.raises(ValueError):
        queue.validate_plan(tmp_path)

