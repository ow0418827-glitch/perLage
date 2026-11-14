"""重选洗发水组 Ambient-Coop 隔离生成与 189 共享池单条目替换评测流水线单元测试。"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image
import pytest

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

import tools.run_perlage_ambient_coop_shampoo_reselect_20261112 as runner
from src.agent.memory.shared_pool import load_shared_pool
from src.visual._perlage_ambient_coop import AmbientCoopLatentConfig, build_source_public_key
from src.visual.audit import digest, freeze_json, read_json
from src.visual.image_utils import sha256_file
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
    """在 tmp_path 中构造完整通过校验的伪前缀与伪 ambient_coop 冻结目录，用于快速测试评测/复算/报告全链路。"""
    run = (tmp_path / "mock_run").resolve()
    monkeypatch.setattr(runner, "output_root", lambda p: Path(p).resolve())
    plan = runner.prepare(run)
    item = plan["generation_inputs"][runner.GROUP_KEY]
    definition = item["definition"]

    src_rgb = np.asarray(Image.open(definition["data"]["source"]).convert("RGB"))
    h, w = src_rgb.shape[:2]

    # 1. 构造有效四阶段 PerLAGE_no_pixel 前缀
    p_method = runner.prefix_method_dir(run)
    p_gen = p_method / "generation"
    seg_src = p_gen / "foreground_segmentation/source"
    seg_tgt = p_gen / "foreground_segmentation/target"
    stats_dir = p_gen / "foreground_statistics"
    init_dir = p_gen / "initialization"
    harm_dir = p_gen / "color_harmonization"
    lat_dir = p_gen / "latent_generation_refinement"
    for d in (seg_src, seg_tgt, stats_dir, init_dir / "attack", harm_dir, lat_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 创建非零前景掩码（保留 ROI 之外像素以便 halo_error/ambient 合法）
    fg_arr = np.zeros((h, w), dtype=np.uint8)
    fg_arr[20 : h - 20, 15 : w - 15] = 255
    fg_path = seg_src / "foreground.png"
    Image.fromarray(fg_arr, mode="L").save(fg_path)
    tgt_fg_path = seg_tgt / "foreground.png"
    Image.new("L", (64, 64), 255).save(tgt_fg_path)

    seg_src_rec = seg_src / "record.json"
    seg_tgt_rec = seg_tgt / "record.json"
    freeze_json(seg_src_rec, {"identity": {"boxes": definition["source_boxes"]}})
    freeze_json(seg_tgt_rec, {"identity": {"boxes": definition["target_boxes"]}})

    body_path = stats_dir / "body_samples.png"
    _make_png(body_path, (100, 110, 90))
    stats_json = stats_dir / "statistics.json"
    freeze_json(stats_json, {"rgb": [100, 110, 90]})

    # init files
    gen_only_png = _make_png(init_dir / "attack/generative_anchor.png", (200, 200, 200), size=(w, h))
    nat_final_png = _make_png(init_dir / "attack/natural_final.png", (200, 200, 200), size=(w, h))
    edit_mask_gen = _make_png(init_dir / "generation/edit_mask.png", (255, 255, 255), size=(w, h))
    init_id = init_dir / "run_identity.json"
    freeze_json(
        init_id,
        {
            "training_queries": definition["data"]["training_queries"],
            "surface_regions": definition["print_regions"],
            "public_binding": {"shadow_pool_used": False},
        },
    )
    opt_id = init_dir / "attack/identity.json"
    freeze_json(opt_id, {"config": {"pixel_clean_steps": 0, "pixel_diffusion_steps": 0, "pixel_levels": 0}})
    init_ready = init_dir / "ready.json"
    freeze_json(
        init_ready,
        {
            "status": "completed",
            "candidates": {
                "generation_only": {"image": str(gen_only_png), "sha256": sha256_file(gen_only_png)},
                "natural_final": {"image": str(nat_final_png), "sha256": sha256_file(nat_final_png)},
            },
        },
    )

    # harm & prefix final (直接用原图拷贝确保 SSIM/MAE/ROI 严格通过门槛)
    harm_anchor = harm_dir / "generative_anchor.png"
    harm_initial = harm_dir / "initial.png"
    harm_mask = harm_dir / "edit_mask.png"
    harm_meta = harm_dir / "metadata.json"
    Image.fromarray(src_rgb).save(harm_anchor)
    Image.fromarray(src_rgb).save(harm_initial)
    Image.fromarray(np.where(fg_arr > 0, 255, 0).astype(np.uint8), mode="L").save(harm_mask)
    freeze_json(harm_meta, {"status": "completed"})

    p_adv = p_gen / "adversarial.png"
    p_anch = p_gen / "generative_anchor.png"
    Image.fromarray(src_rgb).save(p_adv)
    Image.fromarray(src_rgb).save(p_anch)
    p_lat_meta = lat_dir / "metadata.json"
    freeze_json(
        p_lat_meta,
        {
            "status": "completed",
            "config": {"steps": 60},
            "history": [{"step": 60}],
            "best": {"step": 60, "halo_error": 0.0, "selection": [1, 0.5]},
            "accepted_updates": 60,
            "rejected_updates": 0,
        },
    )
    p_lat_id = lat_dir / "identity.json"
    freeze_json(p_lat_id, {"method": "PerLAGE_no_pixel"})

    p_method_id = p_gen / "method_identity.json"
    config_digest = digest(definition)
    p_ready_stages = {
        "foreground_segmentation": {
            "status": "completed",
            "files": {str(p): sha256_file(p) for p in (seg_src_rec, fg_path, seg_tgt_rec, tgt_fg_path)},
        },
        "foreground_statistics": {
            "status": "completed",
            "files": {str(p): sha256_file(p) for p in (stats_json, body_path)},
        },
        "color_harmonization": {
            "status": "completed",
            "files": {str(p): sha256_file(p) for p in (harm_meta, harm_anchor, harm_initial, harm_mask)},
        },
        "latent_generation_refinement": {
            "status": "completed",
            "steps_completed": 60,
            "steps_requested": 60,
            "files": {str(p): sha256_file(p) for p in (p_lat_meta, p_lat_id, p_adv, p_anch)},
        },
    }
    source_sha = sha256_file(definition["data"]["source"])
    target_sha = sha256_file(definition["data"]["target"])
    import hashlib
    src_title_sha = hashlib.sha256(definition["data"]["source_title"].encode("utf-8")).hexdigest()
    freeze_json(
        p_method_id,
        {
            "method": "PerLAGE_no_pixel",
            "definition": "perlage_no_pixel_foreground_statistics_harmonization_perceptual_latent_v1",
            "configuration_sha256": config_digest,
            "source_sha256": source_sha,
            "target_sha256": target_sha,
        },
    )
    p_ready = {
        "method": "PerLAGE_no_pixel",
        "definition": "perlage_no_pixel_foreground_statistics_harmonization_perceptual_latent_v1",
        "status": "completed",
        "color_scheme": "statistics",
        "training_data_policy": "catalog_image_and_public_title_only_v1",
        "shadow_pool_used": False,
        "purchase_history_used": False,
        "victim_memory_used": False,
        "formal_queries_used": False,
        "configuration_sha256": config_digest,
        "method_identity": str(p_method_id),
        "method_identity_sha256": sha256_file(p_method_id),
        "source_sha256": source_sha,
        "target_sha256": target_sha,
        "source_title_sha256": src_title_sha,
        "candidates": {"final": {"image": str(p_adv), "sha256": sha256_file(p_adv)}},
        "initialization_identity": str(init_id),
        "initialization_optimizer_identity": str(opt_id),
        "initialization_evidence": {str(p): sha256_file(p) for p in (init_id, opt_id, init_ready)},
        "stages": p_ready_stages,
        "pixel_optimization": False,
        "auxiliary_max_levels": 0,
        "latent_config": {"auxiliary_levels": 0},
        "optimization_mode": "latent_only",
    }
    freeze_json(p_gen / "ready.json", p_ready)
    freeze_json(
        p_gen / "completion.json",
        {
            "method": "PerLAGE_no_pixel",
            "status": "success",
            "output_image": str(p_adv),
            "output_image_sha256": sha256_file(p_adv),
            "source_image_sha256": source_sha,
            "target_image_sha256": target_sha,
            "perlage_ready": p_ready,
            "shadow_pool_used": False,
        },
    )
    p_frozen_png = p_method / "frozen.png"
    shutil.copyfile(p_adv, p_frozen_png)
    freeze_json(
        p_method / "FROZEN_IMAGE.json",
        {
            "status": "frozen",
            "image": str(p_frozen_png),
            "image_sha256": sha256_file(p_frozen_png),
        },
    )

    # 2. 构造有效 ambient_coop 成图目录
    a_dir = runner.ambient_dir(run)
    a_dir.mkdir(parents=True, exist_ok=True)
    # 仅在背景区域轻微调暗 3 级，保证 ROI 0 改动、前景 <= 6、背景 <= 28
    amb_rgb = src_rgb.copy().astype(np.int16)
    bg_mask = ~(fg_arr > 0)
    amb_rgb[bg_mask] = np.clip(amb_rgb[bg_mask] - 3, 0, 255)
    amb_u8 = amb_rgb.astype(np.uint8)
    a_adv = a_dir / "adversarial.png"
    a_anch = a_dir / "generative_anchor.png"
    Image.fromarray(amb_u8).save(a_adv)
    Image.fromarray(amb_u8).save(a_anch)
    a_mask = _make_png(a_dir / "latent_edit_mask.png", (128, 128, 128), size=(w, h))
    a_audit = a_dir / "public_title_encoding_audit.json"
    freeze_json(a_audit, {"ok": True})
    a_meta = a_dir / "metadata.json"
    freeze_json(
        a_meta,
        {
            "status": "completed",
            "steps_completed": 350,
            "best": {"step": 260},
            "accepted_updates": 350,
            "rejected_updates": 0,
            "fg_latent_max_levels": 0,
            "fg_latent_mae_levels": 0.0,
            "bg_latent_max_levels": 3,
            "bg_latent_mae_levels": 3.0,
            "final_check": {
                "source_ssim": 0.99,
                "source_mae_levels": 1.2,
                "fg_source_mae_levels": 0.0,
                "bg_source_mae_levels": float(np.abs(amb_u8.astype(np.float32) - src_rgb.astype(np.float32))[bg_mask].mean()),
                "fg_anchor_mae_levels": 0.0,
                "bg_anchor_mae_levels": float(np.abs(amb_u8.astype(np.float32) - src_rgb.astype(np.float32))[bg_mask].mean()),
                "bg_mean_rgb_shift_levels": [-3.0, -3.0, -3.0],
                "ambient_global_levels": [-3.0, -3.0, -3.0],
                "halo_error": 0.0,
                "joint_scores": [[0.75]],
            },
        },
    )

    prefix_files = [
        p_gen / "ready.json",
        init_id,
        opt_id,
        init_ready,
        harm_anchor,
        harm_initial,
        p_adv,
        nat_final_png,
        gen_only_png,
        harm_mask,
        fg_path,
    ]
    prefix_binding = {
        "root": str(p_gen.resolve()),
        "ready_sha256": sha256_file(p_gen / "ready.json"),
        "perlage_no_pixel_final_sha256": sha256_file(p_adv),
        "generative_anchor_sha256": sha256_file(harm_anchor),
        "anchor_source": "perlage_final",
        "selected_anchor_sha256": sha256_file(p_adv),
        "first_three_stages_reused": True,
        "fresh_first_three_stages_executed": False,
        "initialization_reused": True,
        "ambient_coop_latent_refinement": True,
        "stages": {k: copy.deepcopy(p_ready_stages[k]) for k in ("foreground_segmentation", "foreground_statistics", "color_harmonization")},
        "initialization_evidence": p_ready["initialization_evidence"],
        "files": {str(Path(p).resolve()): sha256_file(p) for p in prefix_files},
    }
    own_key = build_source_public_key(item["source_public_record"], definition["data"]["source_title"])
    a_id_path = a_dir / "method_identity.json"
    freeze_json(
        a_id_path,
        {
            "method": "PerLAGE_ambient_coop",
            "definition": "perlage_ambient_coop_subject_micro_light_background_ambient_smooth_v1",
            "profile": runner.PROFILE,
            "config": plan["profile_configs"][runner.PROFILE],
            "public_definition": definition,
            "source_public_record": item["source_public_record"],
            "source_public_key_text": own_key,
            "prefix_binding": prefix_binding,
            "inputs": {
                str(Path(definition["data"]["source"]).resolve()): source_sha,
                str(Path(definition["data"]["target"]).resolve()): target_sha,
            },
            "pixel_optimization": False,
            "auxiliary_levels": 0,
            "diffusion_calls": 0,
            "halo_hard_gate": False,
            "mask_policy": "perlage_final_subject_micro_light_background_range_centered_ambient_smooth_v1",
        },
    )
    a_stages = copy.deepcopy(prefix_binding["stages"])
    for rec in a_stages.values():
        rec["execution_mode"] = "reused_frozen_prefix"
    a_stages["latent_generation_refinement"] = {
        "status": "completed",
        "steps_completed": 350,
        "steps_requested": 350,
        "execution_mode": "fresh_full_refinement",
        "files": {str(p.resolve()): sha256_file(p) for p in (a_id_path, a_meta, a_adv, a_anch, a_mask, a_audit)},
    }
    freeze_json(
        a_dir / "ready.json",
        {
            "method": "PerLAGE_ambient_coop",
            "definition": "perlage_ambient_coop_subject_micro_light_background_ambient_smooth_v1",
            "profile": runner.PROFILE,
            "status": "completed",
            "color_scheme": "statistics",
            "training_data_policy": "catalog_image_and_public_title_only_v1",
            "first_three_stages_reused": True,
            "fresh_first_three_stages_executed": False,
            "initialization_reused": True,
            "previous_final_refinement_reused": False,
            "method_identity": str(a_id_path.resolve()),
            "method_identity_sha256": sha256_file(a_id_path),
            "source_sha256": source_sha,
            "target_sha256": target_sha,
            "source_title_sha256": src_title_sha,
            "stages": a_stages,
            "latent_config": plan["profile_configs"][runner.PROFILE],
            "source_public_key_text": own_key,
            "candidates": {"final": {"image": str(a_adv.resolve()), "sha256": sha256_file(a_adv)}},
            "generative_anchor": str(a_anch.resolve()),
            "prefix_binding": prefix_binding,
            "pixel_optimization": False,
            "auxiliary_max_levels": 0,
            "optimization_mode": "latent_only",
            "victim_memory_used": False,
            "formal_queries_used": False,
            "shadow_pool_used": False,
            "true_rank_used": False,
            "purchase_history_used": False,
            "diffusion_calls": 0,
        },
    )
    runner.freeze_images(run)
    return run


def test_output_root_rejects_readonly_and_external_dirs():
    for forbidden in (
        runner.BASE,
        runner.RESELECT_DIR,
        runner.HIST_AMBIENT_RUN,
        runner.HIST_NO_PIXEL_RUN,
        runner.PROJECT / "tmp_run",
        runner.PROJECT,
    ):
        with pytest.raises(ValueError):
            runner.output_root(forbidden)
    ok = runner.output_root("output/perlage_ambient_coop_shampoo_reselect_20261112")
    assert ok == (runner.PROJECT / "output/perlage_ambient_coop_shampoo_reselect_20261112").resolve()


def test_prepare_reads_only_public_selection_and_matches_proposed_pool_key(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "output_root", lambda p: Path(p).resolve())
    run = tmp_path / "prep_run"
    plan = runner.prepare(run)
    prepared = read_json(run / "PREPARED.json")
    assert prepared["formal_queries_read"] is False
    assert prepared["shared_pool_read"] is False
    assert prepared["proposed_pool_entry_read"] is False
    assert prepared["shopping_enabled"] is False
    assert plan["source_id"] == "B09B63YYCW"
    assert plan["target_id"] == "B00BDJODWI"
    assert plan["shopping_enabled"] is False

    # 验证训练公开全文键与评估拟替换条目的检索文本键严格一字不差
    item = plan["generation_inputs"]["07_shampoo"]
    trained_key = build_source_public_key(item["source_public_record"], item["definition"]["data"]["source_title"])
    proposed = read_json(runner.RESELECT_COMP / "proposed_pool_error_entry.json")
    assert memory_index_text(proposed, "product_feature") == trained_key


def test_build_isolated_banks_replaces_only_07_shampoo_in_real_189_pool(tmp_path):
    entries = load_shared_pool(runner.BASE / "shared_memory/shared_memory.json")
    proposed = read_json(runner.RESELECT_COMP / "proposed_pool_error_entry.json")
    base_img = _make_png(tmp_path / "baseline.png", (10, 20, 30))
    amb_img = _make_png(tmp_path / "ambient_coop.png", (40, 50, 60))

    banks = runner.build_isolated_banks(
        entries,
        runner.BASE / "shared_memory/shared_memory.json",
        proposed,
        {
            "baseline": {"image": str(base_img), "sha256": sha256_file(base_img)},
            "ambient_coop": {"image": str(amb_img), "sha256": sha256_file(amb_img)},
        },
        resolve_paths=True,
    )
    assert set(banks) == {"clean", "baseline", "ambient_coop"}
    for name in ("clean", "baseline", "ambient_coop"):
        assert len(banks[name]) == 189
        assert len({e["Id"] for e in banks[name]}) == 189

    # 检查 clean 中旧洗发水被移除、其余 188 条顺序及内容不变
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
    assert sum(e["RecordKind"] == "purchase_memory" for e in banks["clean"]) == sum(
        e["RecordKind"] == "purchase_memory" for e in entries
    )
    assert sum(e["RecordKind"] == "simulated_gallery" for e in banks["clean"]) == sum(
        e["RecordKind"] == "simulated_gallery" for e in entries
    )

    assert all("RetrievalImage" not in e for e in banks["clean"])
    for mod in ("baseline", "ambient_coop"):
        replaced = [e["Id"] for e in banks[mod] if "RetrievalImage" in e]
        assert replaced == [runner.NEW_SHAMPOO_MEMORY_ID]
        assert [{k: v for k, v in e.items() if k != "RetrievalImage"} for e in banks[mod]] == banks["clean"]


def test_build_isolated_banks_rejects_collisions_and_fabricated_purchase(tmp_path):
    entries = load_shared_pool(runner.BASE / "shared_memory/shared_memory.json")
    proposed = read_json(runner.RESELECT_COMP / "proposed_pool_error_entry.json")
    img = _make_png(tmp_path / "img.png", (1, 2, 3))
    assets = {
        "baseline": {"image": str(img), "sha256": sha256_file(img)},
        "ambient_coop": {"image": str(img), "sha256": sha256_file(img)},
    }

    bad_obs = copy.deepcopy(proposed)
    bad_obs["ObservedPurchase"] = True
    with pytest.raises(ValueError):
        runner.build_isolated_banks(entries, runner.BASE / "shared_memory/shared_memory.json", bad_obs, assets)

    bad_coll = copy.deepcopy(proposed)
    bad_coll["FinalASIN"] = entries[0]["FinalASIN"]
    with pytest.raises(ValueError):
        runner.build_isolated_banks(entries, runner.BASE / "shared_memory/shared_memory.json", bad_coll, assets)

    with pytest.raises(ValueError):
        runner.build_isolated_banks(entries[:188], runner.BASE / "shared_memory/shared_memory.json", proposed, assets)


def test_incomplete_generation_blocks_before_reading_pool_or_queries(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "output_root", lambda p: Path(p).resolve())
    run = tmp_path / "inc_run"
    runner.prepare(run)

    orig_read = runner.read_json

    def guarded_read(path):
        if str(path).endswith(("shared_memory.json", "queries.json", "proposed_pool_error_entry.json")):
            raise AssertionError(f"未冻结完整成图时严禁读取评估输入：{path}")
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
    assert cert["rankings"] == 120
    assert cert["score_count"] == 22680

    # 模拟不同子进程 PID 执行 verify
    orig_pid = cert["pid"]
    monkeypatch.setattr(runner.os, "getpid", lambda: orig_pid + 9999)
    runner.verify(run)
    ver = read_json(run / "INDEPENDENT_VERIFICATION.json")
    assert ver["status"] == "passed"
    assert ver["rankings_recomputed"] == 120
    assert ver["score_comparisons"] == 22680
    assert ver["max_score_error"] <= 1e-6

    # verify 同 PID 应被拒绝（删除证书重测）
    (run / "INDEPENDENT_VERIFICATION.json").unlink()
    monkeypatch.setattr(runner.os, "getpid", lambda: orig_pid)
    with pytest.raises(ValueError):
        runner.verify(run)

    # 恢复不同 PID 复算并通过，随后生成 report
    monkeypatch.setattr(runner.os, "getpid", lambda: orig_pid + 9999)
    runner.verify(run)
    summary = runner.report(run)
    assert summary["status"] == "completed"
    assert summary["shopping_executed"] is False
    assert (run / "REPORT.md").is_file()
    assert (run / "SUMMARY.json").is_file()
    assert (run / "comparison_visual_07_shampoo.png").is_file()
    report_text = (run / "REPORT.md").read_text(encoding="utf-8")
    assert "B09B63YYCW" in report_text and "22680" in report_text
