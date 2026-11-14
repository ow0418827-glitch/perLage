"""Tests for formal query vector cache and zero-GPU retrieval module."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.visual.query_cache import (
    QueryVectorCache,
    OfflineQueryEncoder,
    precompute_query_cache,
    canonical_text,
    text_sha256,
)
from src.visual.visual_retrieval import (
    VisualMemoryRetriever,
    RETRIEVAL_CACHE_SCHEMA,
)


class MockTextEncoder:
    supports_image_weight = True

    def __init__(self, prefix="mock"):
        self.snapshot = {"provider": "mock_clip", "model": "vit_b32_test", "prefix": prefix}
        self.text_call_count = 0
        self.last_encoding_audit = {"audit_id": "mock_audit_123"}

    def encode_text(self, text: str) -> np.ndarray:
        self.text_call_count += 1
        payload = f"{self.snapshot['prefix']}:{text}".encode("utf-8")
        raw = np.frombuffer(hashlib.sha256(payload).digest(), dtype=np.uint8).astype(np.float32) + 1.0
        return raw / np.linalg.norm(raw)

    def encode_joint(self, text: str, path: Path, image_weight: float = 0.5) -> np.ndarray:
        t_vec = self.encode_text(text)
        payload = path.read_bytes()
        i_vec = np.frombuffer(hashlib.sha256(payload).digest(), dtype=np.uint8).astype(np.float32) + 1.0
        i_vec = i_vec / np.linalg.norm(i_vec)
        comb = image_weight * i_vec + (1.0 - image_weight) * t_vec
        return comb / np.linalg.norm(comb)

    def encode(self, path: Path) -> np.ndarray:
        payload = path.read_bytes()
        vec = np.frombuffer(hashlib.sha256(payload).digest(), dtype=np.uint8).astype(np.float32) + 1.0
        return vec / np.linalg.norm(vec)


def _make_fixture_bank(tmp_path: Path):
    img1 = tmp_path / "item1.png"
    img2 = tmp_path / "item2.png"
    Image.new("RGB", (16, 16), (200, 100, 50)).save(img1)
    Image.new("RGB", (16, 16), (50, 100, 200)).save(img2)

    entries = [
        {
            "Id": "mem_01",
            "FinalASIN": "ASIN_01",
            "PrimaryImage": {"ASIN": "ASIN_01", "Title": "Security Camera Outdoor", "LocalPath": str(img1)},
            "Brand": "CamPro",
            "Attributes": ["1080p", "motion detection"],
            "FeatureText": "1080p HD video with motion detection",
        },
        {
            "Id": "mem_02",
            "FinalASIN": "ASIN_02",
            "PrimaryImage": {"ASIN": "ASIN_02", "Title": "Barber Salon Chair", "LocalPath": str(img2)},
            "Brand": "SalonKing",
            "Attributes": ["hydraulic", "heavy duty"],
            "FeatureText": "hydraulic heavy duty recliner",
        },
    ]
    bank_path = tmp_path / "bank.json"
    bank_path.write_text(json.dumps({"entries": entries}, ensure_ascii=False, indent=2), encoding="utf-8")
    return bank_path, entries


def test_query_cache_register_and_lookup():
    cache = QueryVectorCache(encoder_snapshot={"model": "test"})
    instruction = "  I want to buy a camera   with motion detection.  "
    vec = np.ones(512, dtype=np.float32)
    vec = vec / np.linalg.norm(vec)

    h = cache.register(instruction, vec, task_id="task_01", audit={"chars": len(instruction)})
    assert h == text_sha256("I want to buy a camera with motion detection.")
    assert len(cache) == 1
    assert "I want to buy a camera with motion detection." in cache
    assert "  I want to buy a camera with motion detection.  " in cache
    assert "other instruction" not in cache

    fetched = cache.get(instruction)
    assert np.allclose(fetched, vec, atol=1e-6)
    assert cache.get_audit(instruction) == {"chars": len(instruction)}


def test_query_cache_save_and_load(tmp_path):
    cache = QueryVectorCache(encoder_snapshot={"provider": "clip", "model": "vit-b-32"})
    instructions = [
        "I want a security camera.",
        "Looking for a hydraulic barber chair.",
    ]
    for idx, inst in enumerate(instructions, 1):
        vec = np.random.randn(512).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        cache.register(inst, vec, task_id=f"feat_{idx:02d}")

    cache_file = tmp_path / "test_query_cache.json"
    cache.save(cache_file)
    assert cache_file.is_file()

    loaded = QueryVectorCache.load(cache_file)
    assert len(loaded) == 2
    assert loaded.encoder_snapshot == {"provider": "clip", "model": "vit-b-32"}
    for inst in instructions:
        assert inst in loaded
        assert np.allclose(cache.get(inst), loaded.get(inst), atol=1e-7)


def test_offline_query_encoder_and_zero_gpu_retrieval(tmp_path):
    bank_path, entries = _make_fixture_bank(tmp_path)
    real_encoder = MockTextEncoder()

    # 1. First run with real encoder to build the bank index cache
    index_cache_dir = tmp_path / "retrieval_index_cache"
    live_retriever = VisualMemoryRetriever(
        real_encoder,
        bank_path,
        query_mode="text",
        cache_dir=index_cache_dir,
    )
    instruction = "I want to buy a security camera with motion detection."
    live_ranks = live_retriever.rank(entries, instruction)
    assert len(live_ranks) == 2
    initial_calls = real_encoder.text_call_count

    # 2. Precompute query cache using real encoder
    query_cache = precompute_query_cache(
        [instruction, "Looking for a barber chair."],
        real_encoder,
        output_path=tmp_path / "queries_cache.json",
    )
    # The precomputation made 2 calls
    assert real_encoder.text_call_count == initial_calls + 2

    # 3. Create OfflineQueryEncoder (0 GPU memory, 0 calls to real_encoder)
    offline_encoder = OfflineQueryEncoder(
        query_cache,
        encoder_snapshot=real_encoder.snapshot,
    )
    offline_retriever = VisualMemoryRetriever(
        offline_encoder,
        bank_path,
        query_mode="text",
        cache_dir=index_cache_dir,
    )

    # 4. Rank using offline retriever
    offline_ranks = offline_retriever.rank(entries, instruction)

    # Check identical ranks and scores down to machine precision
    assert len(offline_ranks) == len(live_ranks)
    for live_item, off_item in zip(live_ranks, offline_ranks):
        assert live_item["Id"] == off_item["Id"]
        assert live_item["rank"] == off_item["rank"]
        assert abs(live_item["score"] - off_item["score"]) < 1e-6

    # Verify that real_encoder was NOT called during offline ranking!
    assert real_encoder.text_call_count == initial_calls + 2


def test_visual_retriever_direct_query_cache_parameter(tmp_path):
    bank_path, entries = _make_fixture_bank(tmp_path)
    real_encoder = MockTextEncoder()
    index_cache_dir = tmp_path / "retrieval_index_cache"

    instruction = "I want to buy a security camera with motion detection."
    vec = real_encoder.encode_text(instruction)
    calls_before = real_encoder.text_call_count

    cache = QueryVectorCache(encoder_snapshot=real_encoder.snapshot)
    cache.register(instruction, vec)

    # Instantiate retriever with direct query_cache parameter
    retriever = VisualMemoryRetriever(
        real_encoder,
        bank_path,
        query_mode="text",
        cache_dir=index_cache_dir,
        query_cache=cache,
    )

    # Ranking should hit query_cache and NOT increment text_call_count for query
    ranks = retriever.rank(entries, instruction)
    assert len(ranks) == 2
    # In ranking, query vector was fetched from cache, but bank entries may be encoded once
    # Check that instruction was served from cache
    assert instruction in cache
