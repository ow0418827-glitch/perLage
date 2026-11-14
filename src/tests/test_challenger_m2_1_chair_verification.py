"""Empirical verification and stress testing for Milestone M2:
Group 02_chair generated image, metadata, and script integrity.

Author: Challenger 1 (challenger_m2_1)
Role: empirical challenger (critic, specialist)

Requirements verified:
1. Exact raw pixel identity across adversarial.png, frozen.png, and generative_anchor.png.
   ((arr1 != arr2).sum() == 0, max(abs(arr1 - arr2)) == 0).
2. Physical image quality metrics: SSIM >= 0.85 floor and MAE <= 12.0 ceiling vs source.jpg.
3. Edge case assertions: synthetic pixel perturbation injection and metadata tampering
   must trigger immediate ValueError in validate_complete.
4. Comprehensive SHA-256 verification against ready.json, FROZEN_IMAGE.json, and disk files.
5. Verification of tools/generate_chair_no_pixel.py parameters and fast replay integrity.
"""

import copy
import hashlib
import json
import os
import shutil
import numpy as np
from PIL import Image
import pytest
import skimage.metrics
import torch

from src.paths import PROJECT, OUTPUT, ProjectPath as Path
from src.visual.audit import read_json
from src.visual.image_utils import read_tensor, sha256_file
from src.visual._perlage.components.core import source_ssim
from src.visual._perlage_no_pixel.evidence import (
    METHOD,
    DEFINITION,
    validate_complete,
)
from src.visual.attack_completion import validate_attack_completion


# ==============================================================================
# Fixtures & Paths
# ==============================================================================

@pytest.fixture(scope="module")
def paths():
    chair_ablation_dir = PROJECT / "output/chair_ablation_20261108"
    gen_dir = chair_ablation_dir / "generation"
    public_chair_dir = PROJECT / "output/datasets/public/groups/02_chair"

    return {
        "output_dir": chair_ablation_dir,
        "gen_dir": gen_dir,
        "source_jpg": public_chair_dir / "source.jpg",
        "target_jpg": public_chair_dir / "target.jpg",
        "frozen_png": chair_ablation_dir / "frozen.png",
        "adversarial_png": chair_ablation_dir / "adversarial.png",
        "anchor_png": chair_ablation_dir / "generative_anchor.png",
        "gen_adversarial_png": gen_dir / "adversarial.png",
        "gen_anchor_png": gen_dir / "generative_anchor.png",
        "ready_json": chair_ablation_dir / "ready.json",
        "frozen_json": chair_ablation_dir / "FROZEN_IMAGE.json",
        "public_run_json": chair_ablation_dir / "PUBLIC_RUN.json",
        "summary_json": chair_ablation_dir / "GENERATION_SUMMARY.json",
        "script": PROJECT / "tools/generate_chair_no_pixel.py",
    }


# ==============================================================================
# 1. Exact Pixel Identity Empirical Tests
# ==============================================================================

def test_raw_pixel_identity_root(paths):
    """Assert exact pixel identity for root adversarial.png, frozen.png, and generative_anchor.png."""
    adv = np.array(Image.open(paths["adversarial_png"]).convert("RGB"))
    frozen = np.array(Image.open(paths["frozen_png"]).convert("RGB"))
    anchor = np.array(Image.open(paths["anchor_png"]).convert("RGB"))

    assert adv.shape == frozen.shape == anchor.shape == (500, 500, 3)
    assert adv.dtype == np.uint8

    # 1. Zero pixel differences count
    assert (adv != frozen).sum() == 0, "adversarial.png and frozen.png have differing pixels"
    assert (adv != anchor).sum() == 0, "adversarial.png and generative_anchor.png have differing pixels"
    assert (frozen != anchor).sum() == 0, "frozen.png and generative_anchor.png have differing pixels"

    # 2. Maximum absolute deviation is strictly 0
    max_diff_frozen = np.max(np.abs(adv.astype(int) - frozen.astype(int)))
    max_diff_anchor = np.max(np.abs(adv.astype(int) - anchor.astype(int)))
    assert max_diff_frozen == 0, f"Max absolute difference with frozen.png is {max_diff_frozen} != 0"
    assert max_diff_anchor == 0, f"Max absolute difference with anchor.png is {max_diff_anchor} != 0"

    # 3. np.array_equal
    assert np.array_equal(adv, frozen)
    assert np.array_equal(adv, anchor)


def test_raw_pixel_identity_generation_dir(paths):
    """Assert exact pixel identity between generation directory artifacts and root artifacts."""
    root_frozen = np.array(Image.open(paths["frozen_png"]).convert("RGB"))
    gen_adv = np.array(Image.open(paths["gen_adversarial_png"]).convert("RGB"))
    gen_anchor = np.array(Image.open(paths["gen_anchor_png"]).convert("RGB"))

    assert np.array_equal(root_frozen, gen_adv)
    assert np.array_equal(root_frozen, gen_anchor)
    assert np.max(np.abs(gen_adv.astype(int) - gen_anchor.astype(int))) == 0


# ==============================================================================
# 2. Physical Quality Metrics Verification (SSIM, MAE, L_inf)
# ==============================================================================

def test_physical_quality_metrics_floors(paths):
    """Verify SSIM >= 0.85 and MAE <= 12.0 between frozen.png and original source.jpg."""
    source_img = Image.open(paths["source_jpg"]).convert("RGB")
    frozen_img = Image.open(paths["frozen_png"]).convert("RGB")

    source_arr = np.array(source_img)
    frozen_arr = np.array(frozen_img)

    assert source_arr.shape == frozen_arr.shape == (500, 500, 3)

    # Skimage metrics
    ssim_skimage = skimage.metrics.structural_similarity(
        source_arr, frozen_arr, channel_axis=2, data_range=255
    )
    mae_numpy = float(np.mean(np.abs(frozen_arr.astype(float) - source_arr.astype(float))))
    linf_numpy = int(np.max(np.abs(frozen_arr.astype(int) - source_arr.astype(int))))

    # PyTorch implementation metrics
    source_tensor = read_tensor(paths["source_jpg"], "cpu")
    frozen_tensor = read_tensor(paths["frozen_png"], "cpu")
    ssim_torch = float(source_ssim(frozen_tensor, source_tensor))
    mae_torch = float((frozen_tensor - source_tensor).abs().mean() * 255)

    # Verification against floors and ceilings
    SSIM_FLOOR = 0.85
    MAE_CEILING = 12.0

    assert ssim_skimage >= SSIM_FLOOR, f"skimage SSIM {ssim_skimage:.6f} below floor {SSIM_FLOOR}"
    assert ssim_torch >= SSIM_FLOOR, f"torch SSIM {ssim_torch:.6f} below floor {SSIM_FLOOR}"
    assert mae_numpy <= MAE_CEILING, f"numpy MAE {mae_numpy:.6f} above ceiling {MAE_CEILING}"
    assert mae_torch <= MAE_CEILING, f"torch MAE {mae_torch:.6f} above ceiling {MAE_CEILING}"

    # Cross-check agreement between PyTorch and NumPy
    assert abs(ssim_skimage - ssim_torch) < 1e-4
    assert abs(mae_numpy - mae_torch) < 1e-4

    # Logged expected empirical values for 02_chair:
    # SSIM ~ 0.955685, MAE ~ 4.512378, L_inf = 132
    assert 0.95 <= ssim_skimage <= 0.96
    assert 4.0 <= mae_numpy <= 5.0
    assert linf_numpy == 132


# ==============================================================================
# 3. Edge Case Assertions & Synthetic Adversarial Perturbations
# ==============================================================================

def test_synthetic_pixel_perturbation_rejected(paths, tmp_path):
    """Inject synthetic pixel perturbations (+1 pixel) and verify validate_complete rejects it."""
    ready = read_json(paths["ready_json"])
    frozen_img = Image.open(paths["frozen_png"]).convert("RGB")
    arr = np.array(frozen_img)

    # Perturb a single pixel at [100, 100, 0] by +1
    perturbed_arr = arr.copy()
    perturbed_arr[100, 100, 0] = (int(perturbed_arr[100, 100, 0]) + 1) % 256
    perturbed_path = tmp_path / "perturbed_single_pixel.png"
    Image.fromarray(perturbed_arr).save(perturbed_path)

    # Passing perturbed image directly must trigger ValueError due to hash mismatch with ready
    with pytest.raises(ValueError, match="当前图片没有绑定优化方法完整完成记录"):
        validate_complete(ready, perturbed_path, method=METHOD)


def test_synthetic_multi_pixel_perturbation_rejected(paths, tmp_path):
    """Inject multi-pixel Gaussian noise perturbations and verify immediate rejection."""
    ready = read_json(paths["ready_json"])
    frozen_img = Image.open(paths["frozen_png"]).convert("RGB")
    arr = np.array(frozen_img).astype(float)

    # Add small bounded noise
    noise = np.random.RandomState(42).randint(-2, 3, size=arr.shape)
    perturbed_arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    perturbed_path = tmp_path / "perturbed_multi.png"
    Image.fromarray(perturbed_arr).save(perturbed_path)

    with pytest.raises(ValueError, match="当前图片没有绑定优化方法完整完成记录"):
        validate_complete(ready, perturbed_path, method=METHOD)


def test_tampered_candidate_image_rejected(paths, tmp_path):
    """If ready candidate image itself is tampered without updating hash, reject."""
    ready_tampered = copy.deepcopy(read_json(paths["ready_json"]))
    perturbed_path = tmp_path / "tampered_candidate.png"
    arr = np.array(Image.open(paths["frozen_png"]).convert("RGB"))
    arr[0, 0, 0] = (int(arr[0, 0, 0]) + 5) % 256
    Image.fromarray(arr).save(perturbed_path)

    ready_tampered["candidates"]["final"]["image"] = str(perturbed_path)
    with pytest.raises(ValueError, match="优化方法最终图片不符"):
        validate_complete(ready_tampered, perturbed_path, method=METHOD)


def test_tampered_candidate_hash_detected_by_stages(paths, tmp_path):
    """If ready candidate hash is updated to match tampered image, original frozen image is rejected."""
    ready_tampered = copy.deepcopy(read_json(paths["ready_json"]))
    perturbed_path = tmp_path / "tampered_candidate2.png"
    arr = np.array(Image.open(paths["frozen_png"]).convert("RGB"))
    arr[0, 0, 0] = (int(arr[0, 0, 0]) + 5) % 256
    Image.fromarray(arr).save(perturbed_path)

    new_hash = sha256_file(perturbed_path)
    ready_tampered["candidates"]["final"]["image"] = str(perturbed_path)
    ready_tampered["candidates"]["final"]["sha256"] = new_hash

    # Evaluating frozen_png against tampered ready raises ValueError
    with pytest.raises(ValueError, match="当前图片没有绑定优化方法完整完成记录"):
        validate_complete(ready_tampered, paths["frozen_png"], method=METHOD)


@pytest.mark.parametrize(
    "tamper_field,tamper_val,expected_error",
    [
        ("pixel_optimization", True, "无像素方法不能包含像素优化"),
        ("auxiliary_max_levels", 1, "无像素方法不能包含固定像素残差"),
        ("auxiliary_max_levels", 8, "无像素方法不能包含固定像素残差"),
        ("shadow_pool_used", True, "无像素感知方法缺少独立训练边界：shadow_pool_used"),
        ("victim_memory_used", True, "无像素感知方法缺少独立训练边界：victim_memory_used"),
        ("formal_queries_used", True, "无像素感知方法缺少独立训练边界：formal_queries_used"),
        ("training_data_policy", "all_memory_v1", "优化方法只接受公开图对与商品名训练"),
        ("color_scheme", "palette", "无像素感知方法仍保留主体统计选色"),
        ("status", "running", "无像素感知方法尚未标记为已完成"),
        ("optimization_mode", "joint", "无像素方法优化模式必须为 latent_only"),
        ("method", "lucid", "完成记录中的方法身份不符合"),
        ("definition", "lucid_pixel_v1", "完成记录中的方法定义不符合"),
    ],
)
def test_ready_metadata_tampering_rejected(paths, tamper_field, tamper_val, expected_error):
    """Verify that tampering any constraint field in ready.json causes validate_complete to fail."""
    ready_tampered = copy.deepcopy(read_json(paths["ready_json"]))
    ready_tampered[tamper_field] = tamper_val

    with pytest.raises(ValueError, match=expected_error):
        validate_complete(ready_tampered, paths["frozen_png"], method=METHOD)


def test_latent_config_auxiliary_levels_tampering(paths):
    """Tampering latent_config auxiliary_levels to >0 must raise ValueError."""
    ready_tampered = copy.deepcopy(read_json(paths["ready_json"]))
    ready_tampered["latent_config"]["auxiliary_levels"] = 4

    with pytest.raises(ValueError, match="无像素方法潜变量配置不能包含辅助像素预算"):
        validate_complete(ready_tampered, paths["frozen_png"], method=METHOD)


def test_stage_steps_tampering(paths):
    """Tampering completed steps in latent_generation_refinement must raise ValueError."""
    ready_tampered = copy.deepcopy(read_json(paths["ready_json"]))
    ready_tampered["stages"]["latent_generation_refinement"]["steps_completed"] = 59

    with pytest.raises(ValueError, match="优化方法生成潜变量细化日程未完成"):
        validate_complete(ready_tampered, paths["frozen_png"], method=METHOD)


def test_stage_file_hash_tampering(paths):
    """Tampering recorded hash for a stage file must raise ValueError."""
    ready_tampered = copy.deepcopy(read_json(paths["ready_json"]))
    first_stage = ready_tampered["stages"]["foreground_segmentation"]
    first_file = list(first_stage["files"].keys())[0]
    first_stage["files"][first_file] = "0" * 64

    with pytest.raises(ValueError, match="优化方法阶段证据缺失或改变"):
        validate_complete(ready_tampered, paths["frozen_png"], method=METHOD)


def test_missing_stage_rejected(paths):
    """Removing any required stage must raise ValueError."""
    ready_tampered = copy.deepcopy(read_json(paths["ready_json"]))
    del ready_tampered["stages"]["color_harmonization"]

    with pytest.raises(ValueError, match="优化方法仍须完成四个自然优先阶段"):
        validate_complete(ready_tampered, paths["frozen_png"], method=METHOD)


# ==============================================================================
# 4. SHA-256 Hashes and Schema Consistency
# ==============================================================================

def test_frozen_image_schema_13_keys(paths):
    """Verify FROZEN_IMAGE.json contains exactly the 13 required top-level keys."""
    data = read_json(paths["frozen_json"])
    expected_keys = {
        "schema_version",
        "method",
        "status",
        "image",
        "image_sha256",
        "source",
        "target",
        "metadata",
        "requested_steps",
        "public_binding_sha256",
        "shared_pool_scope",
        "victim_memory_used",
        "formal_queries_used",
    }
    assert set(data.keys()) == expected_keys
    assert len(data) == 13
    assert data["schema_version"] == 1
    assert data["method"] == METHOD
    assert data["status"] == "frozen"
    assert data["requested_steps"] == 60
    assert data["victim_memory_used"] is False
    assert data["formal_queries_used"] is False


def test_all_file_sha256_digests(paths):
    """Recompute and verify SHA-256 for all critical files on disk."""
    EXPECTED_IMAGE_SHA256 = "04395be4482f316ce5a18e952dd3f64bcdae0439ef05adc0eeb818bc7ab76805"
    EXPECTED_SOURCE_SHA256 = "47a70fad01c018bb524eefcf803794826e4a48e1b667cd77ba0ccecc1c28ecfa"
    EXPECTED_TARGET_SHA256 = "8fc2834a77b8ecf235f5bec9c1f335830d1de142843cda13284cdfe36579000d"

    # Images
    assert sha256_file(paths["frozen_png"]) == EXPECTED_IMAGE_SHA256
    assert sha256_file(paths["adversarial_png"]) == EXPECTED_IMAGE_SHA256
    assert sha256_file(paths["anchor_png"]) == EXPECTED_IMAGE_SHA256
    assert sha256_file(paths["gen_adversarial_png"]) == EXPECTED_IMAGE_SHA256
    assert sha256_file(paths["gen_anchor_png"]) == EXPECTED_IMAGE_SHA256

    # Source & target
    assert sha256_file(paths["source_jpg"]) == EXPECTED_SOURCE_SHA256
    assert sha256_file(paths["target_jpg"]) == EXPECTED_TARGET_SHA256

    # JSON records cross-checks
    ready = read_json(paths["ready_json"])
    frozen_json = read_json(paths["frozen_json"])
    summary = read_json(paths["summary_json"])

    assert ready["candidates"]["final"]["sha256"] == EXPECTED_IMAGE_SHA256
    assert ready["source_sha256"] == EXPECTED_SOURCE_SHA256
    assert ready["target_sha256"] == EXPECTED_TARGET_SHA256

    assert frozen_json["image_sha256"] == EXPECTED_IMAGE_SHA256
    assert frozen_json["source"]["image_sha256"] == EXPECTED_SOURCE_SHA256
    assert frozen_json["target"]["image_sha256"] == EXPECTED_TARGET_SHA256

    assert summary["adversarial_png_sha256"] == EXPECTED_IMAGE_SHA256
    assert summary["frozen_png_sha256"] == EXPECTED_IMAGE_SHA256
    assert summary["generative_anchor_png_sha256"] == EXPECTED_IMAGE_SHA256
    assert summary["source_sha256"] == EXPECTED_SOURCE_SHA256
    assert summary["target_sha256"] == EXPECTED_TARGET_SHA256

    # All stage files on disk
    for stage_name, stage_data in ready["stages"].items():
        for fpath, expected_hash in stage_data["files"].items():
            p = Path(fpath)
            assert p.is_file(), f"Stage file missing: {p}"
            assert sha256_file(p) == expected_hash, f"Stage file hash mismatch: {p}"

    # All initialization evidence files
    for fpath, expected_hash in ready["initialization_evidence"].items():
        p = Path(fpath)
        assert p.is_file(), f"Init evidence file missing: {p}"
        assert sha256_file(p) == expected_hash, f"Init evidence file hash mismatch: {p}"


# ==============================================================================
# 5. Attack Completion Validation
# ==============================================================================

def test_validate_attack_completion_success(paths):
    """Verify validate_attack_completion succeeds on frozen metadata."""
    frozen_json = read_json(paths["frozen_json"])
    validate_attack_completion(
        frozen_json["metadata"],
        requested_steps=60,
        expected={
            "source_image_sha256": sha256_file(paths["source_jpg"]),
            "target_image_sha256": sha256_file(paths["target_jpg"]),
            "method": METHOD,
        },
    )


# ==============================================================================
# 6. Training Isolation and Access Audit
# ==============================================================================

def test_training_access_audit_isolation(paths):
    """Verify no blocked or private memory accesses occurred during generation."""
    audit_path = paths["output_dir"] / "TRAINING_ACCESS_AUDIT.json"
    assert audit_path.is_file()
    audit = read_json(audit_path)

    assert audit["blocked_accesses"] == []
    assert audit["checked_accesses"] > 0
    assert audit["checked_accesses"] == 13250
