"""重选洗发水组 Lucid (8/255, 1000步) 隔离生成与 189 共享池单条目替换评测流水线单元测试。"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image
import pytest

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

import tools.run_lucid_shampoo_reselect_20261112 as runner
from src.agent.memory.shared_pool import load_shared_pool
from src.visual._perlage_ambient_coop import build_source_public_key
from src.visual.audit import freeze_json, read_json
from src.visual.image_utils import sha256_file
from src.visual.io import decoded_image_audit
from src.visual.visual_retrieval import memory_index_text


class _DummyEncoder:
    supports_image_weight = True

    def __init__(self, device="cpu"):
        self.device = device
        self.last_encoding_audit = {"encoder": "dummy"}
        self.snapshot = {
            "provider": "dummy",
            "name": "DummyClip",
            "pretrained": "unit_test",
            "image_weight_supported": True,
        }

    def _vec(self, key: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(key)) % (2**32))
        v = rng.normal(size=16).astype(np.float32)
        return v / np.linalg.norm(v)

    def encode_text(self, text: str) -> np.ndarray:
        self.last_encoding_audit = {"mode": "text", "chars": len(text)}
        return self._vec("text::" + text)

    def encode_joint(self, text: str, image: str | Path, image_weight: float = 0.5) -> np.ndarray:
        img_hash = sha256_file(image)
        self.last_encoding_audit = {"mode": "joint", "image_sha256": img_hash, "weight": float(image_weight)}
        vt = self._vec("text::" + text)
        vi = self._vec("img::" + img_hash)
        v = float(image_weight) * vi + (1.0 - float(image_weight)) * vt
        return (v / np.linalg.norm(v)).astype(np.float32)

    def close(self) -> None:
        return None


def _make_png(path: Path, color: tuple[int, int, int], size=(32, 32)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


def _build_valid_mock_run(tmp_path: Path, monkeypatch) -> Path:
    """在 tmp_path 中构造完整通过校验的伪 Lucid (8/255, 1000步) 冻结目录，用于快速测试评测/复算/报告全链路。"""
    run = (tmp_path / "mock_lucid_run").resolve()
    monkeypatch.setattr(runner, "output_root", lambda p: Path(p).resolve())
    plan = runner.prepare(run)
    item = plan["generation_inputs"][runner.GROUP_KEY]
    opts = plan["generation_options"]

    src_rgb = np.asarray(Image.open(item["source_image"]).convert("RGB"), dtype=np.int16)
    adv_rgb = np.clip(src_rgb.copy(), 0, 255)
    # 施加不超过 8 级的可控伪扰动以验证 L_inf 审计与 diff 图生成
    adv_rgb[:20, :20] = np.clip(adv_rgb[:20, :20] + 4, 0, 255)
    adv_u8 = adv_rgb.astype(np.uint8)

    m_root = runner.method_dir(run)
    gen_dir = m_root / "generation"
    gen_dir.mkdir(parents=True, exist_ok=True)

    adv_png = gen_dir / "adversarial.png"
    frozen_png = m_root / "frozen.png"
    Image.fromarray(adv_u8).save(adv_png)
    shutil.copyfile(adv_png, frozen_png)

    source_sha = sha256_file(item["source_image"])
    target_sha = sha256_file(item["target_image"])
    disk_audit = decoded_image_audit(item["source_image"], adv_png, maximum_levels=8)

    meta = {
        "schema_version": 2,
        "method": "lucid",
        "status": "success",
        "seed": 42,
        "output_image": str(adv_png.resolve()),
        "output_image_sha256": sha256_file(adv_png),
        "source_image_sha256": source_sha,
        "target_image_sha256": target_sha,
        "attack": {
            "epsilon": 8 / 255,
            "spatial_step_size": 0.5 / 255,
            "reference_steps": 1000,
            "steps_requested": 1000,
            "steps_completed": 1000,
            "stop_reason": "requested_steps_completed",
            "selected_step": 1000,
            "text_weight": 1.0,
        },
        "decoded_image_audit": disk_audit,
        "text_alignment": {
            "enabled": True,
            "source": "target_product_features",
            "text_sha256": hashlib.sha256(opts["target_text"].encode("utf-8")).hexdigest(),
            "weight": 1.0,
        },
        "after_attack_saved": {
            "loss": 1.23,
            "global_similarity": [0.8, 0.75, 0.78],
            "local_similarity": [0.7, 0.7, 0.7],
            "text_similarity": 0.41,
        },
    }
    freeze_json(gen_dir / "generation_result.json", meta)
    freeze_json(
        m_root / "FROZEN_IMAGE.json",
        {
            "schema_version": 1,
            "method": "lucid",
            "status": "frozen",
            "image": str(frozen_png.resolve()),
            "image_sha256": sha256_file(frozen_png),
            "requested_steps": 1000,
            "epsilon_levels": 8,
            "text_weight": 1.0,
        },
    )
    runner.freeze_images(run)
    return run


def test_output_root_rejects_readonly_and_external_dirs():
    for forbidden in (
        runner.BASE,
        runner.RESELECT_DIR,
        runner.HIST_AMBIENT_RESELECT_RUN,
        runner.HIST_AMBIENT_RUN,
        runner.HIST_NO_PIXEL_RUN,
        runner.HIST_LUCID_RUN,
        runner.HIST_SHAMPOO_MODES,
        runner.PROJECT / "tmp_run",
        runner.PROJECT,
    ):
        with pytest.raises(ValueError):
            runner.output_root(forbidden)
    ok = runner.output_root("output/lucid_shampoo_reselect_20261112")
    assert ok == (runner.PROJECT / "output/lucid_shampoo_reselect_20261112").resolve()


def test_prepare_reads_only_public_selection_and_pins_8_255_1000_steps(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "output_root", lambda p: Path(p).resolve())
    run = tmp_path / "prep_lucid_run"
    plan = runner.prepare(run)
    prepared = read_json(run / "PREPARED.json")
    assert prepared["formal_queries_read"] is False
    assert prepared["shared_pool_read"] is False
    assert prepared["proposed_pool_entry_read"] is False
    assert prepared["shopping_enabled"] is False
    assert plan["method"] == "lucid"
    assert plan["generation_options"]["steps"] == 1000
    assert plan["generation_options"]["epsilon_levels"] == 8
    assert plan["generation_options"]["text_weight"] == 1.0
    assert plan["source_id"] == "B09B63YYCW"
    assert plan["target_id"] == "B00BDJODWI"

    item = plan["generation_inputs"]["07_shampoo"]
    trained_key = build_source_public_key(item["source_public_record"], item["public_definition"]["data"]["source_title"])
    proposed = read_json(runner.RESELECT_COMP / "proposed_pool_error_entry.json")
    assert memory_index_text(proposed, "product_feature") == trained_key


def test_build_isolated_banks_replaces_only_07_shampoo_in_real_189_pool(tmp_path):
    entries = load_shared_pool(runner.BASE / "shared_memory/shared_memory.json")
    proposed = read_json(runner.RESELECT_COMP / "proposed_pool_error_entry.json")
    lucid_img = _make_png(tmp_path / "lucid.png", (15, 25, 35))

    banks = runner.build_isolated_banks(
        entries,
        runner.BASE / "shared_memory/shared_memory.json",
        proposed,
        {"lucid": {"image": str(lucid_img), "sha256": sha256_file(lucid_img)}},
        resolve_paths=True,
    )
    assert set(banks) == {"clean", "lucid"}
    for name in ("clean", "lucid"):
        assert len(banks[name]) == 189
        assert len({e["Id"] for e in banks[name]}) == 189

    old_ids = {e["Id"] for e in banks["clean"]}
    assert runner.OLD_SHAMPOO_MEMORY_ID not in old_ids
    assert runner.NEW_SHAMPOO_MEMORY_ID in old_ids

    new_row = next(e for e in banks["clean"] if e["Id"] == runner.NEW_SHAMPOO_MEMORY_ID)
    assert new_row["FinalASIN"] == "B09B63YYCW"
    assert new_row["ObservedPurchase"] is False
    assert new_row["RecordKind"] == "simulated_gallery"
    assert new_row["Completed"] is False and new_row["Actions"] == []
    assert sum(e["RecordKind"] == "purchase_memory" for e in banks["clean"]) == 51
    assert sum(e["RecordKind"] == "simulated_gallery" for e in banks["clean"]) == 138

    assert all("RetrievalImage" not in e for e in banks["clean"])
    replaced = [e["Id"] for e in banks["lucid"] if "RetrievalImage" in e]
    assert replaced == [runner.NEW_SHAMPOO_MEMORY_ID]
    assert [{k: v for k, v in e.items() if k != "RetrievalImage"} for e in banks["lucid"]] == banks["clean"]


def test_build_isolated_banks_rejects_fabricated_purchase_and_collision(tmp_path):
    entries = load_shared_pool(runner.BASE / "shared_memory/shared_memory.json")
    proposed = read_json(runner.RESELECT_COMP / "proposed_pool_error_entry.json")
    img = _make_png(tmp_path / "img.png", (1, 2, 3))
    assets = {"lucid": {"image": str(img), "sha256": sha256_file(img)}}

    bad_obs = copy.deepcopy(proposed)
    bad_obs["ObservedPurchase"] = True
    with pytest.raises(ValueError):
        runner.build_isolated_banks(entries, runner.BASE / "shared_memory/shared_memory.json", bad_obs, assets)

    bad_coll = copy.deepcopy(proposed)
    bad_coll["FinalASIN"] = entries[0]["FinalASIN"]
    with pytest.raises(ValueError):
        runner.build_isolated_banks(entries, runner.BASE / "shared_memory/shared_memory.json", bad_coll, assets)


def test_incomplete_generation_blocks_before_reading_pool_or_queries(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "output_root", lambda p: Path(p).resolve())
    run = tmp_path / "inc_lucid_run"
    runner.prepare(run)

    orig_read = runner.read_json

    def guarded_read(path):
        if str(path).endswith(("shared_memory.json", "queries.json", "proposed_pool_error_entry.json")):
            raise AssertionError(f"未冻结完整 Lucid 成图时严禁读取评估输入：{path}")
        return orig_read(path)

    monkeypatch.setattr(runner, "read_json", guarded_read)
    with pytest.raises(FileNotFoundError):
        runner.evaluate(run)


def test_full_evaluate_verify_report_chain_with_mock_encoder(tmp_path, monkeypatch):
    import src.experiments.shared_evaluation as shared_eval

    run = _build_valid_mock_run(tmp_path, monkeypatch)
    monkeypatch.setattr(shared_eval, "make_eval_encoder", lambda device, cfg=None: _DummyEncoder(device))

    runner.evaluate(run)
    cert = read_json(run / "RETRIEVAL_COMPLETED.json")
    assert cert["rankings"] == 80
    assert cert["score_count"] == 15120

    orig_pid = cert["pid"]
    monkeypatch.setattr(runner.os, "getpid", lambda: orig_pid + 9999)
    runner.verify(run)
    ver = read_json(run / "INDEPENDENT_VERIFICATION.json")
    assert ver["status"] == "passed"
    assert ver["rankings_recomputed"] == 80
    assert ver["score_comparisons"] == 15120
    assert ver["max_score_error"] <= 1e-6

    # verify 同 PID 应被拒绝
    (run / "INDEPENDENT_VERIFICATION.json").unlink()
    monkeypatch.setattr(runner.os, "getpid", lambda: orig_pid)
    with pytest.raises(ValueError):
        runner.verify(run)

    monkeypatch.setattr(runner.os, "getpid", lambda: orig_pid + 9999)
    runner.verify(run)
    summary = runner.report(run)
    assert summary["status"] == "completed"
    assert summary["shopping_executed"] is False
    assert (run / "REPORT.md").is_file()
    assert (run / "SUMMARY.json").is_file()
    assert (run / "comparison_visual_07_shampoo_lucid.png").is_file()
    report_text = (run / "REPORT.md").read_text(encoding="utf-8")
    assert "B09B63YYCW" in report_text and "15120" in report_text
