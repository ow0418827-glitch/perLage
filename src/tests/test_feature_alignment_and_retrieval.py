"""单元测试：验证目标商品特点文字提取、记忆库检索策略升级与三种方法文字对齐统一。"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import pytest

from src.paths import PROJECT, OUTPUT, EXPERIMENT_BASE_GROUPS9
from src.visual.target_features import (
    extract_product_features,
    extract_brand_from_text,
    format_retrieval_text,
    build_natural_buyer_intent,
    CURATED_TARGET_FEATURES,
    CURATED_SOURCE_FEATURES,
)
from src.visual.visual_retrieval import (
    index_text_policy,
    memory_index_text,
    PRODUCT_FEATURE_INDEX_TEXT_POLICY,
    TITLE_INDEX_TEXT_POLICY,
    VisualMemoryRetriever,
)
from src.visual.optimization import generate_attack, normalize_method
from src.experiments.pipeline import generation_definition
from src.visual._perlage.pipeline import validate_definition as validate_perlage
from src.visual._perlage_no_pixel.pipeline import validate_definition as validate_no_pixel


def test_target_features_curated_and_extraction():
    """验证9组目标商品的特点提取与自然意图句式。"""
    for asin, info in CURATED_TARGET_FEATURES.items():
        res = extract_product_features(asin)
        assert res["asin"] == asin
        assert res["brand"] == info["brand"]
        assert len(res["attributes"]) >= 2
        assert res["canonical_sentence"] == info["canonical_sentence"]
        assert len(res["training_queries"]) == 3
        # 句式必须为标准买家自然意图
        assert res["canonical_sentence"].startswith("I want to buy")


def test_source_features_curated_and_extraction():
    """验证9组源商品的核定特点，杜绝单字符数字等劣质属性残片。"""
    for asin, info in CURATED_SOURCE_FEATURES.items():
        res = extract_product_features(asin)
        assert res["asin"] == asin
        assert res["brand"] == info["brand"]
        assert len(res["attributes"]) >= 2
        assert res["canonical_sentence"] == info["canonical_sentence"]
        # 确保不包含单字符数值如 '3'
        assert not any(a.strip() == "3" for a in res["attributes"])


def test_brand_extraction_robustness():
    """验证品牌提取能够正确处理Dr.前缀、量词前缀和by子句。"""
    assert extract_brand_from_text("Dr. Collins Perio Toothbrush, (colors vary) (Pack of 6)") == "Dr. Collins"
    assert extract_brand_from_text("3 Piece Baby Toothbrush Set by Cherish Baby Care - with Infant") == "Cherish Baby Care"
    assert extract_brand_from_text("2) Rockville MS40B 4\" 200 Watt Rollbar Roll Cage Tower Speakers") == "Rockville"
    assert extract_brand_from_text('Outdoor Indoor Security Camera, Enster 5MP HD') == 'Enster'
    assert extract_brand_from_text('PTZ Security Camera Outdoor') == ''
    assert extract_brand_from_text('Windows Treatment Blackout Curtains') == ''


def test_corrected_metadata_never_reintroduces_wrong_brand_or_invented_features():
    dough = extract_product_features('B08Z8MHWVS')
    assert 'Big Dipper' in dough['brand'] and 'Doughlish' not in dough['brand']
    shampoo = extract_product_features('B00BDJODWI')
    assert 'exfoliat' not in json.dumps(shampoo['attributes'] + [shampoo['canonical_sentence']]).lower()
    jeans = extract_product_features('B09HH96NJY')
    assert 'baggy straight-leg' in jeans['canonical_sentence']
    assert 'bell bottom' not in json.dumps(jeans['attributes']).lower()
    source = extract_product_features('B08XYBS678')
    assert 'plant based' in source['attributes']


def test_missing_public_product_fails_instead_of_fabricating_standard_features(tmp_path):
    data = tmp_path / 'metadata.json'
    data.write_text('{"products": {}}', encoding='utf-8')
    with pytest.raises(ValueError, match='不能虚构特点'):
        extract_product_features('NOT_PRESENT', db_path=data)


def test_build_natural_buyer_intent_templates_and_plurals():
    """验证自然买家意图模板的拼装规则及单复数语法处理。"""
    # 单数测试
    cat = "camera"
    attrs = ["optical zoom", "night vision", "motion detection"]
    canonical, queries = build_natural_buyer_intent(cat, attrs)
    assert canonical == "I want to buy a camera with optical zoom and night vision."
    assert len(queries) == 3
    assert "I would like to purchase a camera" in queries[1]
    assert "Looking to buy a camera" in queries[2]

    # 复数测试（不得出现 "a curtains" 或 "a jeans"）
    cat_pl = "curtains"
    attrs_pl = ["blackout drapes", "vintage texture"]
    canonical_pl, _ = build_natural_buyer_intent(cat_pl, attrs_pl)
    assert canonical_pl == "I want to buy curtains with blackout drapes and vintage texture."
    assert "a curtains" not in canonical_pl


def test_format_retrieval_text_excludes_actions():
    """验证记忆库检索文本严格由【商品名称 + 品牌 + 特点】构成，排斥操作流程。"""
    title = "PTZ Outdoor Security Camera"
    brand = "PTZ"
    attrs = ["optical zoom", "motion detection"]
    text = format_retrieval_text(title, brand, attrs)

    assert "Title: PTZ Outdoor Security Camera" in text
    assert "Brand: PTZ" in text
    assert "Features: optical zoom, motion detection" in text

    # 严禁出现操作轨迹相关标记
    for forbidden in ["Action:", "click[", "think[", "Actions and observations", "Reward"]:
        assert forbidden not in text


def test_product_feature_index_policy_and_memory_index_text():
    """验证升级后的 PRODUCT_FEATURE_INDEX_TEXT_POLICY 检索策略。"""
    policy = index_text_policy("product_feature")
    assert policy == PRODUCT_FEATURE_INDEX_TEXT_POLICY
    assert "Brand" in policy["fields"]
    assert "Attributes" in policy["fields"]
    assert "Actions" in policy["excluded"]

    entry = {
        "FinalASIN": "B08V134QWR",
        "PrimaryImage": {
            "ASIN": "B08V134QWR",
            "Title": "PTZ Security Camera",
            "Brand": "PTZ",
            "Attributes": ["optical zoom", "night vision"],
        },
        "Brand": "PTZ",
        "Attributes": ["optical zoom", "night vision"],
        "Actions": ["click[item]"],
    }
    indexed = memory_index_text(entry, mode="product_feature")
    assert "Title: PTZ Security Camera" in indexed
    assert "Brand: PTZ" in indexed
    assert "Features: optical zoom, night vision" in indexed
    assert "click[item]" not in indexed

    # 兼容历史 title 模式
    title_only = memory_index_text(entry, mode="title")
    assert title_only == "PTZ Security Camera"


def test_prepared_experiment_base_structure_and_configs_validity():
    """验证 output/experiment_base_groups9_20261109/ 目录资产完整性与各方法配置有效性。"""
    base_dir = EXPERIMENT_BASE_GROUPS9
    assert base_dir.is_dir(), f"基础准备目录不存在: {base_dir}"

    manifest_file = base_dir / "manifest.json"
    assert manifest_file.is_file(), "缺少根清单 manifest.json"

    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert len(manifest["groups"]) == 9

    # 验证共享记忆库
    pool_file = base_dir / manifest["shared_memory"]["pool_file"]
    assert pool_file.is_file(), "缺少完善后的共享记忆库"
    pool_data = json.loads(pool_file.read_text(encoding="utf-8"))
    assert len(pool_data["entries"]) == 189
    for entry in pool_data["entries"]:
        assert "Brand" in entry
        assert "Attributes" in entry
        assert "FeatureText" in entry

    # 验证检索向量缓存
    cache_file = base_dir / manifest["shared_memory"]["index_cache"]["cache_file"]
    assert cache_file.is_file(), "缺少预计算检索向量缓存"
    cache_data = json.loads(cache_file.read_text(encoding="utf-8"))
    assert len(cache_data["vectors"]) == 189

    # 验证各组定义与独立方法配置文件
    for key, gmeta in manifest["groups"].items():
        gdir = base_dir / gmeta["directory"]
        assert (gdir / "definition.json").is_file()
        assert (gdir / "features.json").is_file()
        assert (gdir / "perlage_config.json").is_file()
        assert (gdir / "perlage_no_pixel_config.json").is_file()
        assert (gdir / "lucid_config.json").is_file()
        assert (gdir / "target.jpg").is_file()
        assert (gdir / "source.jpg").is_file()

        # 核心校验：独立方法配置必须能严格通过各自的 validate_definition 约束
        p_cfg = json.loads((gdir / "perlage_config.json").read_text(encoding="utf-8"))
        np_cfg = json.loads((gdir / "perlage_no_pixel_config.json").read_text(encoding="utf-8"))
        lucid_cfg = json.loads((gdir / "lucid_config.json").read_text(encoding="utf-8"))

        p_obj = validate_perlage(p_cfg)
        assert p_obj.pixel_optimization is True

        np_obj = validate_no_pixel(np_cfg)
        assert np_obj.pixel_optimization is False

        assert lucid_cfg["target_text"] == gmeta["target_feature_text"]
        assert lucid_cfg["text_weight"] == 1.0


def test_visual_memory_retriever_auto_detects_product_feature():
    """验证 VisualMemoryRetriever 在默认配置下自动识别并采用 product_feature 策略。"""
    class MockEncoder:
        snapshot = {"model": "mock"}
        supports_image_weight = True
        def encode_joint(self, text, path, image_weight=0.5):
            import numpy as np
            return np.ones(512)

    mem_path = EXPERIMENT_BASE_GROUPS9 / "shared_memory/shared_memory.json"
    retriever = VisualMemoryRetriever(MockEncoder(), mem_path)
    assert retriever.index_text == "product_feature"
    assert "Brand" in retriever.snapshot["index_text_policy"]["fields"]


def test_pipeline_generation_definition_integrates_feature_text():
    """验证 pipeline.generation_definition 统一将特点文字注入 training_queries。"""
    selection = {
        "source": {"image": "src.jpg", "title": "Source Product", "boxes": [[0, 0, 10, 10]], "text_regions": []},
        "target": {
            "image": "tgt.jpg",
            "title": "Target Product",
            "feature_text": "I want to buy a camera with optical zoom and night vision.",
            "boxes": [[0, 0, 10, 10]],
        },
    }
    defn = generation_definition(selection, {}, method="PerLAGE")
    assert defn["data"]["training_queries"] == ["I want to buy a camera with optical zoom and night vision."]
    assert defn["target_feature_text"] == "I want to buy a camera with optical zoom and night vision."
