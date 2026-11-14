"""Product-feature retrieval exercises the real encoder boundary and disk cache."""
import copy
import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from src.visual.visual_retrieval import (
    VisualMemoryRetriever,
    detect_index_text_policy,
    memory_index_text,
)


class RecordingEncoder:
    supports_image_weight = True

    def __init__(self, model="test_weights_v1"):
        self.snapshot = {"provider": "fixture", "model": model}
        self.calls = []

    def encode_joint(self, text, path, image_weight=0.5):
        self.calls.append(text)
        payload = text.encode("utf-8") + path.read_bytes()
        return np.frombuffer(hashlib.sha256(payload).digest(), dtype=np.uint8).astype(np.float32) + 1

    def encode_text(self, text):
        self.calls.append(('text_only', text))
        return np.frombuffer(hashlib.sha256(text.encode('utf-8')).digest(), dtype=np.uint8).astype(np.float32) + 1


def _bank(tmp_path):
    image = tmp_path / "image.png"
    Image.new("RGB", (8, 8), (60, 90, 120)).save(image)
    entries = [{"Id": f"memory-{i}", "FinalASIN": f"item-{i}",
                "PrimaryImage": {"ASIN": f"item-{i}", "Title": f"Product {i}",
                                 "LocalPath": str(image)},
                "Brand": "Observed Brand", "Attributes": ["washable", "soft"],
                "FeatureText": "washable soft fabric",
                "Actions": ["PRIVATE_ACTION_SENTINEL"],
                "Instruction": "PRIVATE_INSTRUCTION_SENTINEL"} for i in range(2)]
    path = tmp_path / "bank.json"
    path.write_text(json.dumps({"entries": entries}), encoding="utf-8")
    return path, entries, image


def test_product_feature_includes_feature_text_and_excludes_trajectory(tmp_path):
    path, entries, image = _bank(tmp_path)
    encoder = RecordingEncoder()
    retriever = VisualMemoryRetriever(encoder, path, cache_dir=None)
    retriever.rank(entries, "vague request", image)
    assert retriever.index_text == "product_feature"
    assert all("FeatureText: washable soft fabric" in text for text in encoder.calls[1:])
    assert all("PRIVATE_" not in text for text in encoder.calls)


def test_text_only_query_never_requires_a_reference_image(tmp_path):
    path, entries, _ = _bank(tmp_path)
    encoder = RecordingEncoder()
    retriever = VisualMemoryRetriever(encoder, path, query_mode='text', cache_dir=None)
    ranking = retriever.rank(entries, 'I want a washable product.')
    assert len(ranking) == len(entries)
    assert encoder.calls[0] == ('text_only', 'I want a washable product.')
    assert all(row['query_image_sha256'] is None for row in ranking)
    assert all(row['query_image_weight'] == 0 for row in ranking)
    assert all('Title:' in text and 'Features:' in text for text in encoder.calls[1:])


def test_text_only_query_rejects_accidental_image_and_blank_text(tmp_path):
    path, entries, image = _bank(tmp_path)
    retriever = VisualMemoryRetriever(RecordingEncoder(), path, query_mode='text')
    with pytest.raises(ValueError, match='不得附带图片'):
        retriever.rank(entries, 'a product', image)
    with pytest.raises(ValueError, match='不能为空'):
        retriever.rank(entries, '')


def test_auto_policy_detects_features_beyond_first_entry_and_inside_primary(tmp_path):
    path, entries, _ = _bank(tmp_path)
    for entry in entries:
        for field in ("Brand", "Attributes", "FeatureText"):
            entry.pop(field)
    entries[1]["PrimaryImage"]["FeatureText"] = "machine washable"
    path.write_text(json.dumps(entries), encoding="utf-8")
    assert detect_index_text_policy(entries) == "product_feature"
    assert VisualMemoryRetriever(RecordingEncoder(), path).index_text == "product_feature"
    assert "machine washable" in memory_index_text(entries[1], "product_feature")


def test_auto_policy_does_not_hide_invalid_memory_json(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        VisualMemoryRetriever(RecordingEncoder(), path)


def test_real_disk_cache_reuses_bank_vectors_but_invalidates_feature_and_model_changes(tmp_path):
    path, entries, image = _bank(tmp_path)
    first_encoder = RecordingEncoder()
    VisualMemoryRetriever(first_encoder, path).rank(entries, "vague request", image)
    assert len(first_encoder.calls) == 3

    second_encoder = RecordingEncoder()
    VisualMemoryRetriever(second_encoder, path).rank(entries, "vague request", image)
    assert second_encoder.calls == ["vague request"]

    changed = copy.deepcopy(entries)
    changed[0]["FeatureText"] = "different verified fabric"
    path.write_text(json.dumps({"entries": changed}), encoding="utf-8")
    feature_encoder = RecordingEncoder()
    VisualMemoryRetriever(feature_encoder, path).rank(changed, "vague request", image)
    assert len(feature_encoder.calls) == 3
    assert "different verified fabric" in feature_encoder.calls[1]

    model_encoder = RecordingEncoder("test_weights_v2")
    VisualMemoryRetriever(model_encoder, path).rank(changed, "vague request", image)
    assert len(model_encoder.calls) == 3


def test_corrupt_cache_is_disposable(tmp_path):
    path, entries, image = _bank(tmp_path)
    VisualMemoryRetriever(RecordingEncoder(), path).rank(entries, "vague request", image)
    cache = next((tmp_path / "retrieval_index_cache").glob("*.json"))
    cache.write_text("[]", encoding="utf-8")
    encoder = RecordingEncoder()
    VisualMemoryRetriever(encoder, path).rank(entries, "vague request", image)
    assert len(encoder.calls) == 3


def test_image_bytes_change_never_reuses_old_embedding(tmp_path):
    path, entries, image = _bank(tmp_path)
    before = VisualMemoryRetriever(RecordingEncoder(), path).rank(entries, "vague request", image)
    Image.new("RGB", (8, 8), (200, 40, 30)).save(image)
    encoder = RecordingEncoder()
    after = VisualMemoryRetriever(encoder, path).rank(entries, "vague request", image)
    assert len(encoder.calls) == 3
    assert before[0]["image_sha256"] != after[0]["image_sha256"]
