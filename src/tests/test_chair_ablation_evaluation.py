"""Tests for Milestone M3: Group 2 (02_chair) ablation evaluation and independent verification."""
import json
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

from src.paths import PROJECT, OUTPUT
from src.visual.image_utils import sha256_file
from src.visual.audit import read_json


EVAL_DIR = OUTPUT / 'chair_ablation_evaluation_20261108'
REPORT_JSON = EVAL_DIR / 'CHAIR_ABLATION_REPORT.json'
REPORT_MD = EVAL_DIR / 'CHAIR_ABLATION_REPORT.md'

EXPECTED_HASHES = {
    'source': '47a70fad01c018bb524eefcf803794826e4a48e1b667cd77ba0ccecc1c28ecfa',
    'target': '8fc2834a77b8ecf235f5bec9c1f335830d1de142843cda13284cdfe36579000d',
    'historical': '3bd4367cc0f1b3b8064ea405a319d5269dbe9758345bfab799e7358b21f4567f',
    'no_pixel': '04395be4482f316ce5a18e952dd3f64bcdae0439ef05adc0eeb818bc7ab76805',
    'query': '92db4482eb4ea28aed7bec1ab4039b4757525dc854d9a1f191ae03118b1503cf',
    'pool': '5c69c3bdf108ec85db5f5f25a979da70f70dea1b815a0f82d90b37ff1728a4e8',
}


def test_deliverable_files_exist():
    """Verify script and report deliverable files exist."""
    script_path = PROJECT / 'tools/evaluate_chair_ablation.py'
    assert script_path.is_file(), f"Script missing: {script_path}"
    assert REPORT_JSON.is_file(), f"Report JSON missing: {REPORT_JSON}"
    assert REPORT_MD.is_file(), f"Report MD missing: {REPORT_MD}"


def test_input_asset_sha256_hashes():
    """Verify input assets match the frozen SHA256 hashes."""
    paths = {
        'source': OUTPUT / 'datasets/public/groups/02_chair/source.jpg',
        'target': OUTPUT / 'datasets/public/groups/02_chair/target.jpg',
        'historical': OUTPUT / 'shared_groups29_20261104_final/groups/02_chair/methods/PerLAGE/frozen.png',
        'no_pixel': OUTPUT / 'chair_ablation_20261108/frozen.png',
        'query': OUTPUT / 'datasets/private/query_images/02_chair.png',
        'pool': OUTPUT / 'shared_memory/pool_189_v1/shared_memory.json',
    }
    for key, p in paths.items():
        assert p.is_file(), f"File missing: {p}"
        assert sha256_file(p) == EXPECTED_HASHES[key], f"Hash mismatch for {key}"


def test_physical_metrics_verification():
    """Verify physical metrics computed in report match independent recalculation."""
    report = read_json(REPORT_JSON)
    sims = report['physical_similarities']

    source_np = np.asarray(Image.open(OUTPUT / 'datasets/public/groups/02_chair/source.jpg').convert('RGB'))
    hist_np = np.asarray(Image.open(OUTPUT / 'shared_groups29_20261104_final/groups/02_chair/methods/PerLAGE/frozen.png').convert('RGB'))
    no_pixel_np = np.asarray(Image.open(OUTPUT / 'chair_ablation_20261108/frozen.png').convert('RGB'))

    # 1. No pixel vs Source
    ssim_no_pixel = float(structural_similarity(no_pixel_np, source_np, channel_axis=2, data_range=255))
    mae_no_pixel = float(np.mean(np.abs(no_pixel_np.astype(float) - source_np.astype(float))))
    linf_no_pixel = int(np.max(np.abs(no_pixel_np.astype(int) - source_np.astype(int))))

    assert abs(sims['perlage_no_pixel_vs_source']['ssim'] - ssim_no_pixel) < 1e-6
    assert abs(sims['perlage_no_pixel_vs_source']['mae'] - mae_no_pixel) < 1e-4
    assert sims['perlage_no_pixel_vs_source']['linf'] == linf_no_pixel == 132

    # 2. Historical vs Source
    ssim_hist = float(structural_similarity(hist_np, source_np, channel_axis=2, data_range=255))
    mae_hist = float(np.mean(np.abs(hist_np.astype(float) - source_np.astype(float))))
    linf_hist = int(np.max(np.abs(hist_np.astype(int) - source_np.astype(int))))

    assert abs(sims['perlage_historical_vs_source']['ssim'] - ssim_hist) < 1e-6
    assert abs(sims['perlage_historical_vs_source']['mae'] - mae_hist) < 1e-4
    assert sims['perlage_historical_vs_source']['linf'] == linf_hist == 132

    # 3. Assert PerLAGE_no_pixel achieves strictly higher fidelity than historical
    assert ssim_no_pixel > ssim_hist
    assert mae_no_pixel < mae_hist
    assert ssim_no_pixel > 0.95
    assert mae_no_pixel < 4.6


def test_retrieval_ablation_rankings_and_equivalence():
    """Verify retrieval ablation results and behavioral equivalence."""
    report = read_json(REPORT_JSON)
    retrieval = report['retrieval_results']

    # Multimodal mode (0.5 / 0.5)
    joint = retrieval['multimodal_0.5']['conditions']
    assert joint['clean']['mean_rank'] == 3.85
    assert joint['clean']['top3_count'] == 3
    assert joint['clean']['top5_count'] == 20

    # Historical and No Pixel equivalence under multimodal joint retrieval
    assert joint['perlage_historical']['top1_count'] == joint['perlage_no_pixel']['top1_count'] == 0
    assert joint['perlage_historical']['top3_count'] == joint['perlage_no_pixel']['top3_count'] == 20
    assert joint['perlage_historical']['top5_count'] == joint['perlage_no_pixel']['top5_count'] == 20
    assert joint['perlage_historical']['mean_rank'] == joint['perlage_no_pixel']['mean_rank'] == 3.0
    assert abs(joint['perlage_historical']['mrr'] - joint['perlage_no_pixel']['mrr']) < 1e-6

    # Pure image mode (1.0 / 0.0)
    pure_img = retrieval['pure_image_1.0']['conditions']
    assert pure_img['clean']['mean_rank'] == 2.0
    assert pure_img['clean']['top1_count'] == 0
    assert pure_img['clean']['top3_count'] == 20

    # Historical and No Pixel equivalence under pure-image retrieval: both hit 100% Top-1!
    assert pure_img['perlage_historical']['top1_count'] == pure_img['perlage_no_pixel']['top1_count'] == 20
    assert pure_img['perlage_historical']['top3_count'] == pure_img['perlage_no_pixel']['top3_count'] == 20
    assert pure_img['perlage_historical']['top5_count'] == pure_img['perlage_no_pixel']['top5_count'] == 20
    assert pure_img['perlage_historical']['mean_rank'] == pure_img['perlage_no_pixel']['mean_rank'] == 1.0
    assert pure_img['perlage_historical']['mrr'] == pure_img['perlage_no_pixel']['mrr'] == 1.0


def test_independent_verification_certificate():
    """Verify independent verification certificate passed with 0 discrepancy."""
    report = read_json(REPORT_JSON)
    cert = report['independent_verification']

    assert cert['status'] == 'passed'
    assert cert['assertion_zero_discrepancy_passed'] is True
    assert cert['total_candidates_recomputed'] == 22680
    assert cert['max_score_discrepancy'] <= 2e-5
    assert cert['rankings_discrepancy_count'] == 0


def test_report_md_user_convention():
    """Verify Markdown report starts with '公子' and is written in Chinese."""
    text = REPORT_MD.read_text(encoding='utf-8')
    assert text.strip().startswith('# 公子')
    assert '第二组理发椅' in text
    assert '物理保真度' in text
    assert '无像素扰动' in text
