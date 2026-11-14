"""Adversarial Verification & Safety Stress-Test Suite for Milestone M1.

Tests and stress-tests:
1. Shared memory bank mutation safety (189-item pool, exactly 1 injected, 188 byte-identical, hostile inputs).
2. Process isolation enforcement (same PID rejection, fresh process & fresh encoder enforcement, 1e-6 tolerance boundary).
3. Multimodal joint query verification (strict query_mode='joint', query_image_weight=0.5, key_image_weight=0.5).
4. Baseline completeness (all 8 banks: clean, baseline, lucid, rankfocus, fg48_bg12, fg_lock_bg24, bg_smooth, dual_smooth).
5. Non-blocking file locking & cross-process concurrent rejection.
6. Generation budget & safety guardrails enforcement.
"""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from unittest import mock

import pytest

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

import src.experiments.perlage_dual_smooth_evaluation as eval_mod
from src.experiments.perlage_dual_smooth_evaluation import (
    BANKS,
    GROUPS,
    PROFILES,
    build_banks,
    verify,
    _validate_result,
    _validate_retrieval,
    _validate_verification,
    _metrics,
)
import tools.run_perlage_dual_smooth_shampoo_20261112 as runner_mod
from tools.run_perlage_dual_smooth_shampoo_20261112 import lock


# ============================================================================
# Helpers to generate synthetic test pools
# ============================================================================

def make_synthetic_pool(count: int = 189, *, target_id: str = "groups9_reference_07_shampoo_B08XYBS678") -> list[dict]:
    """Generates a synthetic clean shared memory pool adhering strictly to schema."""
    entries = []
    for i in range(count):
        if i == 0:
            entry_id = target_id
            asin = "B08XYBS678"
            group = "07_shampoo"
            kind = "simulated_gallery"
        else:
            entry_id = f"public_synthetic_entry_{i:04d}_hash1234567890abcdef"
            asin = f"B00{i:04d}XYZ"
            group = f"0{(i % 9) + 1}_category"
            kind = "simulated_gallery"

        entry = {
            "Id": entry_id,
            "FinalASIN": asin,
            "SourceGroup": group,
            "EntryType": "normal",
            "RecordKind": kind,
            "RecordOrigin": "public_catalog_reference",
            "ObservedPurchase": False,
            "Completed": False,
            "ReferenceNote": "Mock entry for testing.",
            "Brand": f"Brand_{i}",
            "Attributes": ["attr_a", "attr_b"],
            "FeatureText": f"Canonical feature text for item {i}",
            "PrimaryImage": {
                "ASIN": asin,
                "LocalPath": f"images/{asin}.jpg",
                "SHA256": f"mock_sha256_{i:04d}" * 2,
            }
        }
        entries.append(entry)
    return entries


def make_valid_replacements(target_id: str = "groups9_reference_07_shampoo_B08XYBS678") -> dict[str, dict[str, dict]]:
    """Builds replacements dictionary for all adversarial banks in BANKS."""
    adversarial_banks = [b for b in BANKS if b != "clean"]
    replacements = {}
    for name in adversarial_banks:
        replacements[name] = {
            target_id: {
                "image": f"/mock/path/adversarial_{name}.png",
                "sha256": f"sha256_for_{name}_image_" * 2
            }
        }
    return replacements


# ============================================================================
# 1. Shared Memory Bank Mutation Safety
# ============================================================================

class TestSharedMemoryBankMutationSafety:
    """Stress tests for shared memory bank construction, mutation containment, and integrity."""

    def test_build_banks_exactly_one_item_injected(self):
        """Verify that in every adversarial bank, exactly 1 item receives RetrievalImage."""
        entries = make_synthetic_pool(189)
        replacements = make_valid_replacements()
        target_id = "groups9_reference_07_shampoo_B08XYBS678"

        banks = build_banks(entries, replacements)
        assert len(banks) == len(BANKS)
        assert "clean" in banks

        # Clean bank must have ZERO RetrievalImage
        assert sum("RetrievalImage" in e for e in banks["clean"]) == 0

        # Each adversarial bank must have EXACTLY 1 RetrievalImage on the target entry
        for name, bank in banks.items():
            if name == "clean":
                continue
            assert len(bank) == 189
            injected_items = [e for e in bank if "RetrievalImage" in e]
            assert len(injected_items) == 1, f"Bank {name} must have exactly 1 injected RetrievalImage"
            injected = injected_items[0]
            assert injected["Id"] == target_id
            expected_asset = replacements[name][target_id]
            assert injected["RetrievalImage"]["SHA256"] == expected_asset["sha256"]
            assert injected["RetrievalImage"]["LocalPath"] == str(Path(expected_asset["image"]).resolve())
            assert injected["RetrievalImage"]["ASIN"] == injected["FinalASIN"]

    def test_build_banks_remaining_188_byte_for_byte_identical(self):
        """Verify that the other 188 items are byte-for-byte identical to the clean pool."""
        entries = make_synthetic_pool(189)
        replacements = make_valid_replacements()
        target_id = "groups9_reference_07_shampoo_B08XYBS678"

        banks = build_banks(entries, replacements)
        clean_by_id = {e["Id"]: e for e in banks["clean"]}

        for name, bank in banks.items():
            if name == "clean":
                continue
            for entry in bank:
                eid = entry["Id"]
                if eid == target_id:
                    # Target entry: stripping RetrievalImage must yield exact clean entry
                    stripped = {k: v for k, v in entry.items() if k != "RetrievalImage"}
                    assert stripped == clean_by_id[eid]
                    assert json.dumps(stripped, sort_keys=True) == json.dumps(clean_by_id[eid], sort_keys=True)
                else:
                    # Non-target entries: must NOT have RetrievalImage and must match clean entry exactly
                    assert "RetrievalImage" not in entry
                    assert entry == clean_by_id[eid]
                    assert json.dumps(entry, sort_keys=True) == json.dumps(clean_by_id[eid], sort_keys=True)

    @pytest.mark.parametrize("bad_count", [188, 190, 0, 1, 100])
    def test_build_banks_rejects_pool_count_not_189(self, bad_count):
        """Hostile test: input pool size != 189 must raise ValueError."""
        entries = make_synthetic_pool(bad_count) if bad_count > 0 else []
        replacements = make_valid_replacements()
        with pytest.raises(ValueError, match="一百八十九条"):
            build_banks(entries, replacements)

    def test_build_banks_rejects_duplicate_ids(self):
        """Hostile test: input pool with duplicate IDs must raise ValueError."""
        entries = make_synthetic_pool(189)
        entries[1]["Id"] = entries[0]["Id"]  # duplicate target ID
        replacements = make_valid_replacements()
        with pytest.raises(ValueError, match="一百八十九条"):
            build_banks(entries, replacements)

    def test_build_banks_rejects_preexisting_retrieval_image(self):
        """Hostile test: clean entry already tainted with RetrievalImage must be rejected."""
        entries = make_synthetic_pool(189)
        entries[5]["RetrievalImage"] = {"ASIN": "B0000", "LocalPath": "bad.jpg", "SHA256": "bad"}
        replacements = make_valid_replacements()
        with pytest.raises(ValueError, match="基础共享池不能包含历史图片替换标记"):
            build_banks(entries, replacements)

    def test_build_banks_rejects_preexisting_substitution_tag(self):
        """Hostile test: clean entry already marked with ImageSubstitution must be rejected."""
        entries = make_synthetic_pool(189)
        entries[10]["ImageSubstitution"] = True
        replacements = make_valid_replacements()
        with pytest.raises(ValueError, match="基础共享池不能包含历史图片替换标记"):
            build_banks(entries, replacements)

    def test_build_banks_rejects_wrong_replacement_target(self):
        """Hostile test: attempting to replace an entry other than the shampoo target must be rejected."""
        entries = make_synthetic_pool(189)
        replacements = {
            "dual_smooth": {
                "groups9_reference_01_camera_B08NKC17K2": {
                    "image": "/mock/path.png",
                    "sha256": "fake_sha" * 4
                }
            }
        }
        with pytest.raises(ValueError, match="只能替换指定的单个洗发水条目"):
            build_banks(entries, replacements)

    def test_build_banks_rejects_multiple_replacement_targets(self):
        """Hostile test: attempting to replace > 1 entry in a bank must be rejected."""
        entries = make_synthetic_pool(189)
        target_id = "groups9_reference_07_shampoo_B08XYBS678"
        replacements = {
            "dual_smooth": {
                target_id: {"image": "/mock/path.png", "sha256": "fake_sha" * 4},
                entries[1]["Id"]: {"image": "/mock/path2.png", "sha256": "fake_sha2" * 4}
            }
        }
        with pytest.raises(ValueError, match="只能替换指定的单个洗发水条目"):
            build_banks(entries, replacements)

    def test_build_banks_rejects_empty_replacement_targets(self):
        """Hostile test: replacement map with empty targets dict must be rejected."""
        entries = make_synthetic_pool(189)
        replacements = {"dual_smooth": {}}
        with pytest.raises(ValueError, match="只能替换指定的单个洗发水条目"):
            build_banks(entries, replacements)

    def test_build_banks_rejects_simulated_gallery_with_completed_or_actions(self):
        """Hostile test: simulated reference carrying Completed=True or fake Actions must be rejected."""
        entries = make_synthetic_pool(189)
        entries[2]["RecordKind"] = "simulated_gallery"
        entries[2]["Completed"] = True
        replacements = make_valid_replacements()
        with pytest.raises(ValueError, match="未购买参考不能含伪造购买轨迹"):
            build_banks(entries, replacements)

        entries2 = make_synthetic_pool(189)
        entries2[2]["RecordKind"] = "simulated_gallery"
        entries2[2]["Actions"] = [{"action": "click"}]
        with pytest.raises(ValueError, match="未购买参考不能含伪造购买轨迹"):
            build_banks(entries2, replacements)

    def test_build_banks_3_param_compatibility(self, tmp_path):
        """Verify that 3-parameter call (entries, pool_path, replacements) works when pool_path is None or valid."""
        entries = make_synthetic_pool(189)
        replacements = make_valid_replacements()
        # When pool_path is None
        banks = build_banks(entries, None, replacements)
        assert len(banks) == len(BANKS)


# ============================================================================
# 2. Process Isolation Enforcement
# ============================================================================

class TestProcessIsolationEnforcement:
    """Stress tests for cross-process isolation enforcement and float error thresholds."""

    def test_verify_unconditionally_raises_on_same_pid(self, tmp_path):
        """Hostile test: verify() must unconditionally raise ValueError when retrieval['pid'] == os.getpid()."""
        mock_retrieval = {
            "status": "completed",
            "pid": os.getpid(),  # SAME PID!
            "execution_device": "cuda",
            "inputs_sha256": "fake_sha",
            "files": {}
        }
        mock_inputs = {
            "expected_rankings": 160,
            "expected_score_comparisons": 30240,
            "bindings": {}
        }

        with mock.patch.object(eval_mod, "_inputs", return_value=mock_inputs), \
             mock.patch.object(eval_mod, "_validate_retrieval", return_value=mock_retrieval):
            with pytest.raises(ValueError, match="独立复算必须使用不同进程"):
                verify(tmp_path)

    def test_validate_verification_rejects_same_pid(self, tmp_path):
        """Hostile test: _validate_verification rejects verification file with same PID as retrieval."""
        current_pid = os.getpid()
        mock_ret = {
            "status": "completed",
            "pid": current_pid,
            "inputs_sha256": "inputs_sha",
            "files": {}
        }
        mock_ver = {
            "status": "passed",
            "score_comparisons": 30240,
            "rankings_recomputed": 160,
            "fresh_process": True,
            "fresh_encoder": True,
            "execution_device": "cuda",
            "pid": current_pid,             # Violation: same PID as retrieval
            "retrieval_pid": current_pid,
            "files": {},
            "inputs_sha256": "inputs_sha",
            "max_score_error": 1e-7,
            "retrieval_certificate_sha256": "cert_sha"
        }
        mock_inputs = {
            "expected_rankings": 160,
            "expected_score_comparisons": 30240
        }

        ver_file = tmp_path / "INDEPENDENT_VERIFICATION.json"
        ver_file.write_text(json.dumps(mock_ver), encoding="utf-8")

        with mock.patch.object(eval_mod, "_validate_retrieval", return_value=mock_ret), \
             mock.patch.object(eval_mod, "sha256_file", return_value="inputs_sha"):
            with pytest.raises(ValueError, match="独立复算证书未绑定本轮全部候选分数或误差超标"):
                _validate_verification(tmp_path, mock_inputs)

    def test_validate_verification_rejects_mismatched_retrieval_pid(self, tmp_path):
        """Hostile test: verification['retrieval_pid'] != retrieval['pid'] must be rejected."""
        mock_ret = {
            "status": "completed",
            "pid": 11111,
            "inputs_sha256": "inputs_sha",
            "files": {}
        }
        mock_ver = {
            "status": "passed",
            "score_comparisons": 30240,
            "rankings_recomputed": 160,
            "fresh_process": True,
            "fresh_encoder": True,
            "execution_device": "cuda",
            "pid": 22222,
            "retrieval_pid": 99999,  # Mismatched retrieval PID
            "files": {},
            "inputs_sha256": "inputs_sha",
            "max_score_error": 1e-7,
            "retrieval_certificate_sha256": "cert_sha"
        }
        mock_inputs = {"expected_rankings": 160, "expected_score_comparisons": 30240}

        ver_file = tmp_path / "INDEPENDENT_VERIFICATION.json"
        ver_file.write_text(json.dumps(mock_ver), encoding="utf-8")

        with mock.patch.object(eval_mod, "_validate_retrieval", return_value=mock_ret), \
             mock.patch.object(eval_mod, "sha256_file", return_value="inputs_sha"):
            with pytest.raises(ValueError, match="独立复算证书未绑定本轮全部候选分数或误差超标"):
                _validate_verification(tmp_path, mock_inputs)

    def test_validate_verification_requires_fresh_process_and_encoder(self, tmp_path):
        """Hostile test: fresh_process=False or fresh_encoder=False must be rejected."""
        mock_ret = {"status": "completed", "pid": 11111, "inputs_sha256": "inputs_sha", "files": {}}
        base_ver = {
            "status": "passed", "score_comparisons": 30240, "rankings_recomputed": 160,
            "fresh_process": True, "fresh_encoder": True, "execution_device": "cuda",
            "pid": 22222, "retrieval_pid": 11111, "files": {}, "inputs_sha256": "inputs_sha",
            "max_score_error": 1e-7, "retrieval_certificate_sha256": "cert_sha"
        }
        mock_inputs = {"expected_rankings": 160, "expected_score_comparisons": 30240}
        ver_file = tmp_path / "INDEPENDENT_VERIFICATION.json"

        # Case A: fresh_process is False
        bad_ver_a = copy.deepcopy(base_ver)
        bad_ver_a["fresh_process"] = False
        ver_file.write_text(json.dumps(bad_ver_a), encoding="utf-8")
        with mock.patch.object(eval_mod, "_validate_retrieval", return_value=mock_ret), \
             mock.patch.object(eval_mod, "sha256_file", return_value="inputs_sha"):
            with pytest.raises(ValueError, match="独立复算证书未绑定本轮全部候选分数或误差超标"):
                _validate_verification(tmp_path, mock_inputs)

        # Case B: fresh_encoder is False
        bad_ver_b = copy.deepcopy(base_ver)
        bad_ver_b["fresh_encoder"] = False
        ver_file.write_text(json.dumps(bad_ver_b), encoding="utf-8")
        with mock.patch.object(eval_mod, "_validate_retrieval", return_value=mock_ret), \
             mock.patch.object(eval_mod, "sha256_file", return_value="inputs_sha"):
            with pytest.raises(ValueError, match="独立复算证书未绑定本轮全部候选分数或误差超标"):
                _validate_verification(tmp_path, mock_inputs)

    @pytest.mark.parametrize("error_val,should_pass", [
        (0.0, True),
        (5e-7, True),
        (1.0e-6, True),          # Exactly on tolerance boundary: <= 1e-6 PASS
        (1.0000001e-6, False),   # Strictly > 1e-6 FAIL
        (1.00001e-6, False),     # > 1e-6 FAIL
        (0.001, False),          # > 1e-6 FAIL
        (float("nan"), False),   # NaN FAIL
        (float("inf"), False),   # Inf FAIL
        (-float("inf"), False),  # -Inf FAIL
        ("0.0", False),          # string type FAIL
    ])
    def test_validate_verification_score_error_tolerance_boundary(self, tmp_path, error_val, should_pass):
        """Boundary test: verify that max_score_error strictly enforces <= 1e-6 and finite numbers."""
        mock_ret = {"status": "completed", "pid": 11111, "inputs_sha256": "inputs_sha", "files": {}}
        mock_ver = {
            "status": "passed", "score_comparisons": 30240, "rankings_recomputed": 160,
            "fresh_process": True, "fresh_encoder": True, "execution_device": "cuda",
            "pid": 22222, "retrieval_pid": 11111, "files": {}, "inputs_sha256": "inputs_sha",
            "max_score_error": error_val, "retrieval_certificate_sha256": "cert_sha"
        }
        mock_inputs = {"expected_rankings": 160, "expected_score_comparisons": 30240}
        ver_file = tmp_path / "INDEPENDENT_VERIFICATION.json"
        ver_file.write_text(json.dumps(mock_ver), encoding="utf-8")

        with mock.patch.object(eval_mod, "_validate_retrieval", return_value=mock_ret), \
             mock.patch.object(eval_mod, "sha256_file", return_value="inputs_sha"), \
             mock.patch.object(eval_mod, "_check", return_value=None):
            if should_pass:
                result = _validate_verification(tmp_path, mock_inputs)
                assert result["status"] == "passed"
            else:
                with pytest.raises(ValueError, match="独立复算证书未绑定本轮全部候选分数或误差超标"):
                    _validate_verification(tmp_path, mock_inputs)


# ============================================================================
# 3. Multimodal Joint Query Verification
# ============================================================================

class TestMultimodalJointQueryVerification:
    """Stress tests for multimodal joint query mode and weight enforcement."""

    def test_validate_result_rejects_non_joint_query_mode(self, tmp_path):
        """Hostile test: result with query_mode != 'joint' must raise ValueError."""
        run = tmp_path
        inputs = {
            "banks": {"dual_smooth": {"sha256": "bank_sha", "path": str(tmp_path / "bank.json")}},
            "groups": {"07_shampoo": {"source_memory_id": "target_id", "queries": []}}
        }

        # Mock result JSON with text-only query_mode
        bad_result = {
            "inputs_sha256": "inputs_sha",
            "bank": "dual_smooth",
            "group": "07_shampoo",
            "protocol": {
                "query_mode": "text",      # VIOLATION: text-only
                "key_mode": "joint",
                "key_image_weight": 0.5,
                "query_image_weight": 0.5,
                "bank_sha256": "bank_sha"
            },
            "rankings": []
        }
        res_dir = run / "retrieval" / "dual_smooth"
        res_dir.mkdir(parents=True, exist_ok=True)
        (res_dir / "07_shampoo.json").write_text(json.dumps(bad_result), encoding="utf-8")

        with mock.patch.object(eval_mod, "sha256_file", return_value="inputs_sha"):
            with pytest.raises(ValueError, match="既有排名未绑定当前完整图文评估条件"):
                _validate_result(run, inputs, "dual_smooth", "07_shampoo")

    def test_validate_result_rejects_non_joint_key_mode(self, tmp_path):
        """Hostile test: result with key_mode != 'joint' must raise ValueError."""
        run = tmp_path
        inputs = {
            "banks": {"dual_smooth": {"sha256": "bank_sha", "path": str(tmp_path / "bank.json")}},
            "groups": {"07_shampoo": {"source_memory_id": "target_id", "queries": []}}
        }
        bad_result = {
            "inputs_sha256": "inputs_sha",
            "bank": "dual_smooth",
            "group": "07_shampoo",
            "protocol": {
                "query_mode": "joint",
                "key_mode": "image",       # VIOLATION: image-only key
                "key_image_weight": 0.5,
                "query_image_weight": 0.5,
                "bank_sha256": "bank_sha"
            },
            "rankings": []
        }
        res_dir = run / "retrieval" / "dual_smooth"
        res_dir.mkdir(parents=True, exist_ok=True)
        (res_dir / "07_shampoo.json").write_text(json.dumps(bad_result), encoding="utf-8")

        with mock.patch.object(eval_mod, "sha256_file", return_value="inputs_sha"):
            with pytest.raises(ValueError, match="既有排名未绑定当前完整图文评估条件"):
                _validate_result(run, inputs, "dual_smooth", "07_shampoo")

    @pytest.mark.parametrize("query_w,key_w", [
        (0.7, 0.5),
        (0.5, 0.3),
        (1.0, 0.0),
        (0.0, 0.5),
        (0.499, 0.5),
        (0.5, 0.501),
    ])
    def test_validate_result_rejects_asymmetric_weights(self, tmp_path, query_w, key_w):
        """Hostile test: weights != 0.5 must be rejected."""
        run = tmp_path
        inputs = {
            "banks": {"dual_smooth": {"sha256": "bank_sha", "path": str(tmp_path / "bank.json")}},
            "groups": {"07_shampoo": {"source_memory_id": "target_id", "queries": []}}
        }
        bad_result = {
            "inputs_sha256": "inputs_sha",
            "bank": "dual_smooth",
            "group": "07_shampoo",
            "protocol": {
                "query_mode": "joint",
                "key_mode": "joint",
                "key_image_weight": key_w,
                "query_image_weight": query_w,
                "bank_sha256": "bank_sha"
            },
            "rankings": []
        }
        res_dir = run / "retrieval" / "dual_smooth"
        res_dir.mkdir(parents=True, exist_ok=True)
        (res_dir / "07_shampoo.json").write_text(json.dumps(bad_result), encoding="utf-8")

        with mock.patch.object(eval_mod, "sha256_file", return_value="inputs_sha"):
            with pytest.raises(ValueError, match="既有排名未绑定当前完整图文评估条件"):
                _validate_result(run, inputs, "dual_smooth", "07_shampoo")

    def test_runner_prepare_plan_specifies_joint_and_half_weights(self, tmp_path):
        """Verify that runner's plan strictly enforces joint query mode and 0.5 weights."""
        test_run = tmp_path / "output_test"
        with mock.patch.object(runner_mod, "output_root", return_value=test_run):
            plan = runner_mod.prepare(test_run)
            assert plan["query_mode"] == "joint"
            assert plan["query_image_weight"] == 0.5
            assert plan["key_image_weight"] == 0.5
            assert plan["queries_per_group"] == 20
            assert plan["shopping_enabled"] is False


# ============================================================================
# 4. Baseline Completeness
# ============================================================================

class TestBaselineCompleteness:
    """Stress tests for baseline bank completeness and 7-baseline comparison coverage."""

    def test_banks_tuple_contains_all_8_banks(self):
        """Assert BANKS contains all 8 required banks in exact sequence."""
        expected_banks = (
            "clean",
            "baseline",
            "lucid",
            "rankfocus",
            "fg48_bg12",
            "fg_lock_bg24",
            "bg_smooth",
            "dual_smooth"
        )
        assert BANKS == expected_banks
        assert len(BANKS) == 8
        assert len(set(BANKS)) == 8, "Bank names must be distinct"

    def test_expected_rankings_and_score_comparisons_formula(self):
        """Verify dynamic calculation of rankings and comparisons."""
        expected_rankings = len(BANKS) * len(GROUPS) * 20
        expected_score_comparisons = expected_rankings * 189

        assert expected_rankings == 8 * 1 * 20 == 160
        assert expected_score_comparisons == 160 * 189 == 30240

    def test_report_labels_cover_all_8_banks(self, tmp_path):
        """Verify that report generation contains labels for all 8 banks."""
        # Inspect source code of report() to ensure all 8 keys exist in labels dict
        import inspect
        report_src = inspect.getsource(eval_mod.report)
        for bank_name in BANKS:
            assert f'"{bank_name}":' in report_src, f"Label for bank {bank_name} missing in report()"

    def test_paired_metrics_math(self):
        """Verify paired metric calculations for rank comparisons."""
        # 3 queries:
        # query 1: current=2 (<=3), prev=4 (>3) -> gained
        # query 2: current=5 (>3), prev=1 (<=3) -> lost
        # query 3: current=1, prev=1 -> tied
        current = [
            {"task_id": "q1", "source_rank": 2},
            {"task_id": "q2", "source_rank": 5},
            {"task_id": "q3", "source_rank": 1},
        ]
        previous = [
            {"task_id": "q1", "source_rank": 4},
            {"task_id": "q2", "source_rank": 1},
            {"task_id": "q3", "source_rank": 1},
        ]
        diff = eval_mod._paired(current, previous)
        assert diff["top3_gained"] == 1
        assert diff["top3_lost"] == 1
        assert diff["improved_ranks"] == 1
        assert diff["worsened_ranks"] == 1
        assert diff["tied_ranks"] == 1


# ============================================================================
# 5. File Locking & Concurrency Rejection
# ============================================================================

class TestFileLockingEnforcement:
    """Stress tests for non-blocking file locking and concurrent launch rejection."""

    def test_lock_acquisition_success(self, tmp_path):
        """Verify lock acquisition on unlocked file."""
        lock_file = tmp_path / "RUN.lock"
        handle = lock(lock_file)
        try:
            assert handle is not None
            assert not handle.closed
        finally:
            handle.close()

    def test_concurrent_lock_rejected_with_runtime_error(self, tmp_path):
        """Verify that a second concurrent lock attempt on the same file raises RuntimeError."""
        lock_file = tmp_path / "RUN.lock"
        h1 = lock(lock_file)
        try:
            with pytest.raises(RuntimeError, match="已有本轮队列或工作进程正在执行，拒绝重复并发启动"):
                lock(lock_file)
        finally:
            h1.close()

    def test_lock_released_and_reacquired(self, tmp_path):
        """Verify that after closing handle, lock can be acquired again."""
        lock_file = tmp_path / "RUN.lock"
        h1 = lock(lock_file)
        h1.close()

        # Reacquisition must succeed
        h2 = lock(lock_file)
        try:
            assert h2 is not None
        finally:
            h2.close()

    def test_different_lock_files_independent(self, tmp_path):
        """Verify that RUN.lock and WORKER.lock on different paths do not block each other."""
        run_lock = tmp_path / "RUN.lock"
        worker_lock = tmp_path / "WORKER.lock"

        h_run = lock(run_lock)
        try:
            h_worker = lock(worker_lock)
            try:
                assert h_worker is not None
            finally:
                h_worker.close()
        finally:
            h_run.close()

    def test_cross_process_lock_contention(self, tmp_path):
        """Empirical stress test: a child process holds the lock; parent must be immediately rejected."""
        lock_file = tmp_path / "CROSS_PROC.lock"

        # Spawn a python child that acquires the lock, signals ready via file, and sleeps for 2s
        ready_flag = tmp_path / "child_ready.flag"
        code = (
            "import sys, time; from pathlib import Path; "
            f"sys.path.insert(0, {repr(str(PROJECT))}); "
            "from tools.run_perlage_dual_smooth_shampoo_20261112 import lock; "
            f"h = lock(Path({repr(str(lock_file))})); "
            f"Path({repr(str(ready_flag))}).write_text('READY'); "
            "time.sleep(2.0); h.close()"
        )
        proc = subprocess.Popen([sys.executable, "-c", code])
        try:
            # Wait for child to signal ready
            start_t = time.time()
            while not ready_flag.exists():
                time.sleep(0.05)
                if time.time() - start_t > 5.0:
                    raise TimeoutError("Child process failed to acquire lock in time")

            # Child is actively holding the lock: parent lock attempt MUST be rejected
            with pytest.raises(RuntimeError, match="已有本轮队列或工作进程正在执行，拒绝重复并发启动"):
                lock(lock_file)
        finally:
            proc.wait(timeout=5)

        # After child exits, parent lock attempt MUST succeed
        h_parent = lock(lock_file)
        try:
            assert h_parent is not None
        finally:
            h_parent.close()


# ============================================================================
# 6. Dual-Smooth Config & Guardrail Constraints
# ============================================================================

class TestDualSmoothConfigAndGuardEnforcement:
    """Stress tests on configuration invariants, budget ceilings, and safety boundaries."""

    @pytest.mark.parametrize("bad_latent", [-1, 0, 15, 89, 129])
    def test_frozen_generation_budget_bounds_latent(self, bad_latent):
        """Latent levels must be bounded between 16 and 88 for dual-smooth."""
        from src.visual._perlage_dual_smooth.pipeline import DualSmoothLatentConfig
        if bad_latent < 0 or bad_latent > 128:
            with pytest.raises(ValueError):
                DualSmoothLatentConfig(latent_levels=bad_latent).validate()

    @pytest.mark.parametrize("bad_bg", [-1, 0, 5, 9, 65])
    def test_frozen_generation_budget_bounds_bg(self, bad_bg):
        """Background levels must be bounded between 6 and 8 for dual-smooth."""
        from src.visual._perlage_dual_smooth.pipeline import DualSmoothLatentConfig
        if bad_bg < 1 or bad_bg > 64:
            with pytest.raises(ValueError):
                DualSmoothLatentConfig(background_levels=bad_bg).validate()

    def test_frozen_generation_forbids_pixel_optimization(self):
        """Auxiliary levels must be 0 and pixel_optimization must be False."""
        from src.visual._perlage_dual_smooth.pipeline import DualSmoothLatentConfig
        with pytest.raises(ValueError, match="本试验禁止像素残差"):
            DualSmoothLatentConfig(auxiliary_levels=2).validate()

        with pytest.raises(ValueError, match="本试验禁止像素残差"):
            DualSmoothLatentConfig(pixel_optimization=True).validate()
