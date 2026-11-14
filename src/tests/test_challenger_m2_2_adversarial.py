"""Adversarial stress testing and empirical validation on M2 chair generation deliverables.

This suite is implemented by challenger_m2_2 to verify:
1. Sandbox enforcement and training access audit (zero private memory / formal queries)
2. Live containment behavior of catalog_training_guard
3. Adversarial tampering attacks on FROZEN_IMAGE.json against validate_attack_completion
4. Adversarial tampering attacks on ready.json against validate_complete
5. Physical zero-pixel equivalence (adversarial vs generative_anchor vs frozen)
6. Historical baseline immutability (shared_groups29_20261104_final chair frozen.png)
7. Public input dataset digests and schema compliance
"""
import copy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import pytest

from src.paths import PROJECT, OUTPUT, SHARED_MEMORY, PRIVATE
from src.visual.audit import read_json, digest
from src.visual.image_utils import sha256_file
from src.visual._perlage.components.colour import read_rgb
from src.visual._perlage_no_pixel.evidence import (
    METHOD,
    DEFINITION,
    validate_complete,
)
from src.visual.attack_completion import validate_attack_completion
from src.visual.catalog_training_guard import catalog_training_guard
from src.visual.retrieval_objective import TRAINING_DATA_POLICY


CHAIR_OUTPUT = OUTPUT / "chair_ablation_20261108"
HISTORICAL_BASELINE_PNG = (
    OUTPUT / "shared_groups29_20261104_final/groups/02_chair/methods/PerLAGE/frozen.png"
)
HISTORICAL_BASELINE_SHA256 = (
    "3bd4367cc0f1b3b8064ea405a319d5269dbe9758345bfab799e7358b21f4567f"
)

SOURCE_JPG = OUTPUT / "datasets/public/groups/02_chair/source.jpg"
TARGET_JPG = OUTPUT / "datasets/public/groups/02_chair/target.jpg"
EXPECTED_SOURCE_SHA256 = "47a70fad01c018bb524eefcf803794826e4a48e1b667cd77ba0ccecc1c28ecfa"
EXPECTED_TARGET_SHA256 = "8fc2834a77b8ecf235f5bec9c1f335830d1de142843cda13284cdfe36579000d"
EXPECTED_FROZEN_SHA256 = "04395be4482f316ce5a18e952dd3f64bcdae0439ef05adc0eeb818bc7ab76805"


# ==============================================================================
# 1. Historical Baseline & Public Inputs Integrity
# ==============================================================================

def test_historical_baseline_untouched():
    """Verify that the historical baseline frozen.png was NOT overwritten."""
    assert HISTORICAL_BASELINE_PNG.is_file(), f"Historical baseline not found: {HISTORICAL_BASELINE_PNG}"
    actual_hash = sha256_file(HISTORICAL_BASELINE_PNG)
    assert actual_hash == HISTORICAL_BASELINE_SHA256, (
        f"Historical baseline compromised! Expected {HISTORICAL_BASELINE_SHA256}, got {actual_hash}"
    )


def test_public_chair_inputs_integrity():
    """Verify public chair input pair matches expected canonical hashes."""
    assert SOURCE_JPG.is_file() and TARGET_JPG.is_file()
    assert sha256_file(SOURCE_JPG) == EXPECTED_SOURCE_SHA256
    assert sha256_file(TARGET_JPG) == EXPECTED_TARGET_SHA256


# ==============================================================================
# 2. Sandbox Enforcement & Training Access Audit
# ==============================================================================

def test_training_access_audit_records():
    """Verify TRAINING_ACCESS_AUDIT.json recorded legitimate accesses with 0 blocked accesses."""
    audit_file = CHAIR_OUTPUT / "TRAINING_ACCESS_AUDIT.json"
    assert audit_file.is_file(), "TRAINING_ACCESS_AUDIT.json missing from output root"

    audit_data = read_json(audit_file)
    assert "checked_accesses" in audit_data
    assert "blocked_accesses" in audit_data

    assert isinstance(audit_data["checked_accesses"], int)
    assert audit_data["checked_accesses"] >= 10000, (
        f"Expected at least 10,000 checked accesses, got {audit_data['checked_accesses']}"
    )
    assert audit_data["blocked_accesses"] == [], (
        f"Blocked accesses occurred during training: {audit_data['blocked_accesses']}"
    )

    gen_audit = CHAIR_OUTPUT / "generation/TRAINING_ACCESS_AUDIT.json"
    if gen_audit.is_file():
        assert read_json(gen_audit) == audit_data


def test_catalog_training_guard_active_containment():
    """Adversarially probe catalog_training_guard to prove it actively blocks private access."""
    allowed = [CHAIR_OUTPUT, OUTPUT / "datasets"]

    with catalog_training_guard(allowed_output_roots=allowed) as guard_state:
        # 1. Reading an allowed file succeeds
        allowed_path = CHAIR_OUTPUT / "ready.json"
        if allowed_path.is_file():
            _ = allowed_path.read_text(encoding="utf-8")

        initial_checked = guard_state["checked_accesses"]
        assert initial_checked > 0

        # 2. Attempting to access SHARED_MEMORY must raise PermissionError
        with pytest.raises(PermissionError):
            _ = list(SHARED_MEMORY.iterdir()) if SHARED_MEMORY.exists() else open(SHARED_MEMORY / "fake.json")

        # 3. Attempting to access another output run must raise PermissionError
        with pytest.raises(PermissionError):
            _ = open(HISTORICAL_BASELINE_PNG, "rb").read()

        # 4. Attempting to access forbidden file names must raise PermissionError
        forbidden_test_file = CHAIR_OUTPUT / "queries.json"
        with pytest.raises(PermissionError):
            _ = open(forbidden_test_file, "r")

        # 5. Guard state must record all blocked attempts
        assert len(guard_state["blocked_accesses"]) >= 3


# ==============================================================================
# 3. Physical Zero-Pixel Equivalence Verification
# ==============================================================================

def test_physical_zero_pixel_exact_equality():
    """Verify adversarial.png, frozen.png, and generative_anchor.png are bit-for-bit identical."""
    adv_png = CHAIR_OUTPUT / "adversarial.png"
    frozen_png = CHAIR_OUTPUT / "frozen.png"
    anchor_png = CHAIR_OUTPUT / "generative_anchor.png"

    assert adv_png.is_file() and frozen_png.is_file() and anchor_png.is_file()

    adv_hash = sha256_file(adv_png)
    frozen_hash = sha256_file(frozen_png)
    anchor_hash = sha256_file(anchor_png)

    assert adv_hash == EXPECTED_FROZEN_SHA256
    assert frozen_hash == EXPECTED_FROZEN_SHA256
    assert anchor_hash == EXPECTED_FROZEN_SHA256

    adv_rgb = read_rgb(adv_png)
    frozen_rgb = read_rgb(frozen_png)
    anchor_rgb = read_rgb(anchor_png)

    assert np.array_equal(adv_rgb, anchor_rgb), "Adversarial and anchor RGB arrays differ!"
    assert np.array_equal(adv_rgb, frozen_rgb), "Adversarial and frozen RGB arrays differ!"

    max_diff = np.abs(adv_rgb.astype(np.int32) - anchor_rgb.astype(np.int32)).max()
    assert max_diff == 0, f"Max pixel residual is non-zero: {max_diff}"


# ==============================================================================
# 4. Adversarial Tampering: FROZEN_IMAGE.json & validate_attack_completion
# ==============================================================================

def test_frozen_image_schema_and_keys():
    """Verify FROZEN_IMAGE.json adheres to 13 required top-level keys."""
    frozen_json_path = CHAIR_OUTPUT / "FROZEN_IMAGE.json"
    assert frozen_json_path.is_file()
    data = read_json(frozen_json_path)

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
    assert data["victim_memory_used"] is False
    assert data["formal_queries_used"] is False
    assert data["method"] == METHOD
    assert data["status"] == "frozen"
    assert data["image_sha256"] == EXPECTED_FROZEN_SHA256


@pytest.mark.parametrize(
    "tamper_key,tamper_val,expected_err",
    [
        ("status", "in_progress", "图片方法尚未正常完成"),
        ("output_image_sha256", "0" * 64, "最终图片缺失或摘要改变"),
        ("shadow_pool_used", True, "完整方法不得读入影子池或私有共享记忆池"),
        ("source_image_sha256", "f" * 64, "完成记录未绑定本次冻结输入：source_image_sha256"),
        ("target_image_sha256", "e" * 64, "完成记录未绑定本次冻结输入：target_image_sha256"),
    ],
)
def test_validate_attack_completion_tampering_rejected(tamper_key, tamper_val, expected_err):
    """Adversarially tamper with metadata fields and verify validate_attack_completion rejects them."""
    metadata = copy.deepcopy(read_json(CHAIR_OUTPUT / "completion.json"))
    metadata[tamper_key] = tamper_val

    with pytest.raises(ValueError, match=expected_err):
        validate_attack_completion(
            metadata,
            requested_steps=60,
            expected={
                "source_image_sha256": EXPECTED_SOURCE_SHA256,
                "target_image_sha256": EXPECTED_TARGET_SHA256,
                "method": METHOD,
            },
        )


def test_validate_attack_completion_ready_pair_mismatch():
    """Verify that if expected is omitted, mismatch between metadata and perlage_ready triggers error."""
    metadata = copy.deepcopy(read_json(CHAIR_OUTPUT / "completion.json"))
    metadata["source_image_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="四阶段证据与本次公开图对不符"):
        validate_attack_completion(metadata, requested_steps=60, expected=None)



def test_validate_attack_completion_expected_mismatch():
    """Verify validate_attack_completion rejects mismatch with external expected dictionary."""
    metadata = read_json(CHAIR_OUTPUT / "completion.json")

    with pytest.raises(ValueError, match="完成记录未绑定本次冻结输入：method"):
        validate_attack_completion(
            metadata,
            requested_steps=60,
            expected={
                "source_image_sha256": EXPECTED_SOURCE_SHA256,
                "target_image_sha256": EXPECTED_TARGET_SHA256,
                "method": "lucid",
            },
        )


# ==============================================================================
# 5. Adversarial Tampering: ready.json & validate_complete
# ==============================================================================

@pytest.mark.parametrize(
    "tamper_field,tamper_val,expected_error",
    [
        ("pixel_optimization", True, "无像素方法不能包含像素优化"),
        ("auxiliary_max_levels", 1, "无像素方法不能包含固定像素残差"),
        ("auxiliary_max_levels", 8, "无像素方法不能包含固定像素残差"),
        ("shadow_pool_used", True, "无像素感知方法缺少独立训练边界：shadow_pool_used"),
        ("victim_memory_used", True, "无像素感知方法缺少独立训练边界：victim_memory_used"),
        ("formal_queries_used", True, "无像素感知方法缺少独立训练边界：formal_queries_used"),
        ("purchase_history_used", True, "无像素感知方法缺少独立训练边界：purchase_history_used"),
        ("training_data_policy", "all_memory_v1", "优化方法只接受公开图对与商品名训练"),
        ("color_scheme", "palette", "无像素感知方法仍保留主体统计选色"),
        ("status", "running", "无像素感知方法尚未标记为已完成"),
        ("optimization_mode", "joint", "无像素方法优化模式必须为 latent_only"),
        ("method", "lucid", "完成记录中的方法身份不符合"),
        ("definition", "lucid_pixel_v1", "完成记录中的方法定义不符合"),
    ],
)
def test_ready_zero_pixel_and_boundary_tampering_rejected(tamper_field, tamper_val, expected_error):
    """Verify that tampering any zero-pixel, boundary, or identity parameter triggers validate_complete failure."""
    ready = copy.deepcopy(read_json(CHAIR_OUTPUT / "ready.json"))
    ready[tamper_field] = tamper_val

    with pytest.raises(ValueError, match=expected_error):
        validate_complete(ready, CHAIR_OUTPUT / "frozen.png", method=METHOD)


def test_ready_latent_auxiliary_levels_tampering():
    """Verify setting latent_config auxiliary_levels > 0 fails validation."""
    ready = copy.deepcopy(read_json(CHAIR_OUTPUT / "ready.json"))
    ready["latent_config"]["auxiliary_levels"] = 4

    with pytest.raises(ValueError, match="无像素方法潜变量配置不能包含辅助像素预算"):
        validate_complete(ready, CHAIR_OUTPUT / "frozen.png", method=METHOD)


@pytest.mark.parametrize(
    "deleted_stage",
    [
        "foreground_segmentation",
        "foreground_statistics",
        "color_harmonization",
        "latent_generation_refinement",
    ],
)
def test_ready_missing_stage_rejected(deleted_stage):
    """Verify deleting any of the 4 natural stages causes validate_complete to fail."""
    ready = copy.deepcopy(read_json(CHAIR_OUTPUT / "ready.json"))
    del ready["stages"][deleted_stage]

    with pytest.raises(ValueError, match="优化方法仍须完成四个自然优先阶段"):
        validate_complete(ready, CHAIR_OUTPUT / "frozen.png", method=METHOD)


def test_ready_incomplete_latent_steps_rejected():
    """Verify setting steps_completed < steps_requested fails validation."""
    ready = copy.deepcopy(read_json(CHAIR_OUTPUT / "ready.json"))
    ready["stages"]["latent_generation_refinement"]["steps_completed"] = 59
    ready["stages"]["latent_generation_refinement"]["steps_requested"] = 60

    with pytest.raises(ValueError, match="优化方法生成潜变量细化日程未完成"):
        validate_complete(ready, CHAIR_OUTPUT / "frozen.png", method=METHOD)


def test_ready_corrupted_stage_file_digest_rejected():
    """Verify altering any SHA256 digest in stage file records causes validate_complete to fail."""
    ready = copy.deepcopy(read_json(CHAIR_OUTPUT / "ready.json"))
    seg_files = ready["stages"]["foreground_segmentation"]["files"]
    first_key = next(iter(seg_files))
    seg_files[first_key] = "0" * 64

    with pytest.raises(ValueError, match="优化方法阶段证据缺失或改变"):
        validate_complete(ready, CHAIR_OUTPUT / "frozen.png", method=METHOD)


def test_ready_tampered_candidate_sha256_rejected():
    """Verify changing candidates.final.sha256 fails validation."""
    ready = copy.deepcopy(read_json(CHAIR_OUTPUT / "ready.json"))
    ready["candidates"]["final"]["sha256"] = "a" * 64

    with pytest.raises(ValueError, match="优化方法最终图片不符"):
        validate_complete(ready, CHAIR_OUTPUT / "frozen.png", method=METHOD)


def test_validate_complete_against_wrong_image():
    """Verify passing an altered image file to validate_complete fails."""
    ready = read_json(CHAIR_OUTPUT / "ready.json")
    # SOURCE_JPG is a valid image but not the generated candidate
    with pytest.raises(ValueError, match="当前图片没有绑定优化方法完整完成记录"):
        validate_complete(ready, SOURCE_JPG, method=METHOD)
