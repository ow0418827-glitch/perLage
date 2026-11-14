"""核对特点正文实际进入生成入口和可求导的文字目标，无需启动图片实验。"""
import copy
import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from src.visual.audit import save_json
from src.visual.optimization import generate_attack


def test_lucid_requires_and_forwards_frozen_feature_text(products, tmp_path, monkeypatch):
    import src.visual._lucid.pipeline as lucid
    seen = {}
    def generate(source, target, output, **kwargs):
        seen.update(kwargs)
        return source, dict(status='success')
    monkeypatch.setattr(lucid, 'generate_attack', generate)
    source, target = products[1]['image'], products[0]['image']
    with pytest.raises(ValueError, match='冻结的目标商品特点'):
        generate_attack(source, target, tmp_path/'run', method='lucid')
    text = 'I want to buy shampoo with a gentle formula.'
    save_json(tmp_path/'features.json', dict(canonical_sentence=text))
    generate_attack(source, target, tmp_path/'run', method='lucid')
    assert seen['target_text'] == text and seen['text_weight'] == 1.0
    with pytest.raises(ValueError, match='不一致'):
        generate_attack(source, target, tmp_path/'run2', method='lucid', target_text='Different text')


def test_malformed_features_cannot_silently_disable_text(products, tmp_path):
    (tmp_path/'features.json').write_text('{broken', encoding='utf-8')
    with pytest.raises(json.JSONDecodeError):
        generate_attack(products[1]['image'], products[0]['image'], tmp_path/'run', method='lucid')


def test_perlage_config_relative_images_and_actual_text_are_bound(products, tmp_path, monkeypatch):
    from src.experiments.pipeline import generation_definition
    import src.visual._perlage.pipeline as perlage
    text = 'I want to buy shampoo with a gentle formula.'
    selection = dict(source=products[1], target=dict(products[0], feature_text=text))
    definition = generation_definition(selection, {})
    definition['data']['source'] = 'product1.png'
    definition['data']['target'] = 'product0.png'
    config = tmp_path/'perlage.json'
    save_json(config, definition)
    seen = {}
    def generate(source, target, output, *, definition, device):
        seen.update(definition)
        return Path(source), dict(definition='fixture', stages={})
    monkeypatch.setattr(perlage, 'generate_perlage', generate)
    generate_attack(products[1]['image'], products[0]['image'], tmp_path/'run', perlage_config=config)
    assert seen['data']['source'] == products[1]['image']
    assert seen['data']['target'] == products[0]['image']
    assert seen['data']['training_queries'] == [text]
    invalid = copy.deepcopy(definition)
    invalid['data']['training_queries'] = ['Different text']
    with pytest.raises(ValueError, match='完全一致'):
        generate_attack(products[1]['image'], products[0]['image'], tmp_path/'run2', perlage_config=invalid)


def test_lucid_feature_alignment_contributes_to_pixel_gradient():
    from dataclasses import replace
    from src.visual.encoders.backbone import ImageFeatures
    from src.visual._lucid.config import AttackConfig
    from src.visual._lucid.objective import prepare_targets, joint_objective
    text = 'I want to buy shampoo with a gentle formula.'
    encoded = []
    class Model:
        device = 'cpu'
        cache_identity = 'fixture'
        def encode_image(self, image):
            embedding = F.normalize(image.mean((-2, -1)), dim=-1)
            return ImageFeatures(embedding, embedding[:, None, :].expand(-1, 4, -1))
        def encode_text(self, actual):
            encoded.append(actual)
            return torch.tensor([[1., 0., 0.]])
    models = [Model(), Model(), Model()]
    config = AttackConfig(clusters=2, cluster_iterations=2, sinkhorn_iterations=2, text_weight=1.)
    source = torch.tensor([.2, .5, .7]).reshape(1, 3, 1, 1).expand(-1, -1, 4, 4).clone().requires_grad_(True)
    target = torch.ones_like(source).detach()
    targets = prepare_targets(models, target, text, config, prepare_cluster_centers=False)
    with_text = joint_objective(source, models, targets, torch.ones(3)/3, config, use_cached_target_centers=False)
    without_text = joint_objective(source, models, targets, torch.ones(3)/3, replace(config, text_weight=0.), use_cached_target_centers=False)
    gradient = torch.autograd.grad(with_text.loss - without_text.loss, source)[0]
    assert encoded == [text]
    assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient) > 0
