import copy
from dataclasses import replace
import math
import numpy as np
import pytest

from src.vision.pair_selection import (SelectionConfig, symmetric_patch_similarity,
    score_pair, choose_pair, validate_product)


def features(cosine=1.):
    return dict(cls=np.array([1., 0.]), patches=np.array([[1., 0.], [0., 1.]]),
                text=np.array([cosine, math.sqrt(max(0., 1-cosine*cosine))]),
                clip_image=np.array([1., 0.]), clip_text=np.array([0.7, 0.3]), aspect=1.)


def test_patch_match_is_symmetric_with_unequal_foregrounds():
    a = np.array([[1., 0.], [0., 1.], [0., 1.]])
    b = np.array([[1., 0.]])
    assert symmetric_patch_similarity(a, b) == pytest.approx(symmetric_patch_similarity(b, a))
    assert symmetric_patch_similarity(a, b) == pytest.approx(2/3)


@pytest.mark.parametrize('mutation,reason', [
    ({'id': 'product0'}, 'same_product_id'),
    ({'title': 'PUBLIC product NUMBER 0!!!'}, 'same_normalized_title'),
    ({'owned': False}, 'source_not_attacker_owned')])
def test_explicit_gates(products, mutation, reason):
    source = dict(products[1], **mutation)
    row = score_pair(source, products[0], features(.72), features(), .03)
    assert not row['eligible'] and reason in row['rejected_reasons']


def test_identical_image_is_rejected_even_when_titles_differ(products):
    row = score_pair(dict(products[1], image_sha256=products[0]['image_sha256']), products[0], features(.72), features(), .01)
    assert 'same_image_bytes' in row['rejected_reasons']


def test_cost_is_nonnegative_and_monotone(products):
    low = score_pair(products[1], products[0], features(.72), features(), .01)
    high = score_pair(products[1], products[0], features(.72), features(), .5)
    assert low['eligible'] and low['score'] > high['score']
    with pytest.raises(ValueError):
        score_pair(products[1], products[0], features(.72), features(), float('nan'))


@pytest.mark.parametrize('config', [SelectionConfig(shape_min=float('nan')),
    SelectionConfig(weights=(0., 0., 0., 0.)), SelectionConfig(semantic_min=.9)])
def test_bad_configuration_is_rejected(config):
    with pytest.raises(ValueError):
        config.validate()


def test_no_eligible_candidate_stops_without_fallback(products):
    class Backend:
        snapshot = {'local': True}
        def features(self, record): return features()
        def recolor_cost(self, a, b): return .01
    selection = choose_pair(products[0], products[1:], Backend())
    assert selection['status'] == 'no_eligible_pair'
    assert selection['selected_source_id'] is None


def test_private_fields_and_unreviewed_labels_rejected(products, tmp_path):
    source = dict(products[1])
    source.pop('image_sha256')
    with pytest.raises(ValueError):
        validate_product(dict(source, source_history='购买日志'), tmp_path, owned=True)
    with pytest.raises(ValueError):
        validate_product(dict(source, text_regions_reviewed=False), tmp_path, owned=True)
