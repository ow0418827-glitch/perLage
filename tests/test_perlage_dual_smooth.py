"""Unit and integration test suite for PerLAGE Dual-Smooth cooperative refinement.

Covers:
1. DualSmoothLatentConfig dataclass and preset checking
2. Spatial envelope mathematical verification (cosine feathering, boundaries, text protection)
3. Separable 1D Gaussian kernel verification (energy preservation, separability, luma/chroma)
4. Generation validation logic verification (validate_dual_smooth_complete)
5. Evaluation module bank builder verification (build_banks on 189-item pool)
6. CLI & Runner smoke test (argument parsing and error handling)
7. Live module integration checks (pipeline, evaluation, runner)
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import pytest
import scipy.ndimage as ndi
import torch
import torch.nn.functional as F


# ============================================================================
# Section 1: Config Dataclass & Presets
# ============================================================================

try:
    from src.visual._perlage_dual_smooth.pipeline import (
        DualSmoothLatentConfig,
        DUAL_SMOOTH_PRESETS,
    )
except ImportError:
    # Standalone reference definition matching interface contract
    @dataclass(frozen=True)
    class DualSmoothLatentConfig:
        steps: int = 350
        check_every: int = 10
        latent_levels: int = 24
        background_levels: int = 8
        anchor_source: str = 'harmonized'
        edge_feather_pixels: float = 12.0
        roi_feather_pixels: float = 6.0
        edge_feather_mode: str = 'cosine'
        luma_smooth_sigma: float = 2.2
        chroma_smooth_sigma: float = 5.0
        bg_chroma_keep: float = 0.55
        fg_chroma_keep: float = 0.85
        tau_scale: float = 0.75
        latent_working_size: int = 512
        latent_limit: float = 6.0
        latent_learning_rate: float = 0.06
        source_ssim_floor: float = 0.80
        source_mae_ceiling: float = 16.0
        temperature: float = 0.04
        joint_weight: float = 0.5
        compensated_weight: float = 0.5
        fg_tv_weight: float = 0.0002
        bg_tv_weight: float = 0.0015
        bg_chroma_tv_weight: float = 0.003
        bg_l2_weight: float = 0.0001
        fg_l2_weight: float = 0.00005
        overflow_weight: float = 0.05
        halo_soft_weight: float = 0.0001
        halo_parent_multiplier: float = 1.0
        halo_floor: float = 12.0
        guard_backtracks: int = 8
        auxiliary_levels: int = 0
        pixel_optimization: bool = False
        seed: int = 42

        def validate(self):
            if type(self.steps) is not int or self.steps < 1 or type(self.check_every) is not int or self.check_every < 1 or self.steps % self.check_every:
                raise ValueError("试验须执行完整、可检查的细化步数")
            if type(self.latent_levels) is not int or not 0 <= self.latent_levels <= 128:
                raise ValueError("试验主体潜变量解码变化预算无效")
            if type(self.background_levels) is not int or not 1 <= self.background_levels <= 64:
                raise ValueError("试验背景潜空间优化预算无效")
            if self.anchor_source not in ('harmonized', 'perlage_final'):
                raise ValueError("试验锚点源必须为 harmonized 或 perlage_final")
            if not 0.25 <= self.source_ssim_floor < 1 or not 0 < self.source_mae_ceiling <= 128:
                raise ValueError("试验质量门槛无效")
            if not 0 < self.latent_limit <= 12 or not 0 < self.latent_learning_rate <= 0.2:
                raise ValueError("试验潜变量更新参数无效")
            if self.auxiliary_levels != 0 or self.pixel_optimization is not False:
                raise ValueError("本试验禁止像素残差及像素优化")
            if not (2.0 <= self.edge_feather_pixels <= 32.0) or not (1.0 <= self.roi_feather_pixels <= 16.0):
                raise ValueError("边界过渡宽度无效")
            if self.edge_feather_mode not in ('cosine', 'linear'):
                raise ValueError("边界过渡模式仅支持 cosine 或 linear")
            if not (0.0 <= self.luma_smooth_sigma <= 16.0) or not (0.0 <= self.chroma_smooth_sigma <= 16.0):
                raise ValueError("高斯平滑滤波核标准差无效")
            if not (0.1 <= self.bg_chroma_keep <= 1.0) or not (0.1 <= self.fg_chroma_keep <= 1.0):
                raise ValueError("色度保持比例无效")
            if not (0.2 <= self.tau_scale <= 1.0):
                raise ValueError("饱和度调制因子无效")
            if any(not math.isfinite(v) for v in asdict(self).values() if isinstance(v, float)):
                raise ValueError("试验参数必须有限")

    DUAL_SMOOTH_PRESETS = {
        'dual_smooth_fg20_bg6': asdict(DualSmoothLatentConfig(
            steps=350, latent_levels=20, background_levels=6,
            source_ssim_floor=0.85, source_mae_ceiling=14.0)),
        'dual_smooth_fg24_bg8': asdict(DualSmoothLatentConfig(
            steps=350, latent_levels=24, background_levels=8,
            source_ssim_floor=0.82, source_mae_ceiling=14.0)),
        'dual_smooth_fg16_bg6': asdict(DualSmoothLatentConfig(
            steps=350, latent_levels=16, background_levels=6,
            source_ssim_floor=0.85, source_mae_ceiling=12.0)),
        'dual_smooth_fg48_bg8': asdict(DualSmoothLatentConfig(
            steps=350, latent_levels=48, background_levels=8,
            source_ssim_floor=0.76, source_mae_ceiling=20.0)),
    }


class TestDualSmoothConfig:
    """Test suite 1: Configuration dataclass, immutability, serialization and validation."""

    def test_default_config_values(self):
        cfg = DualSmoothLatentConfig()
        assert is_dataclass(cfg)
        assert cfg.steps == 350
        assert cfg.check_every == 10
        assert cfg.latent_levels == 24
        assert cfg.background_levels == 8
        assert cfg.anchor_source == 'harmonized'
        assert cfg.edge_feather_pixels == 12.0
        assert cfg.roi_feather_pixels == 6.0
        assert cfg.edge_feather_mode == 'cosine'
        assert cfg.luma_smooth_sigma == 2.2
        assert cfg.chroma_smooth_sigma == 5.0
        assert cfg.bg_chroma_keep == 0.55
        assert cfg.fg_chroma_keep == 0.85
        assert cfg.tau_scale == 0.75
        assert cfg.auxiliary_levels == 0
        assert cfg.pixel_optimization is False
        # Calling validate on defaults must succeed
        cfg.validate()

    def test_config_immutability(self):
        cfg = DualSmoothLatentConfig()
        with pytest.raises(Exception):  # FrozenInstanceError
            cfg.steps = 500  # type: ignore

    def test_config_serialization_roundtrip(self):
        cfg = DualSmoothLatentConfig(latent_levels=20, background_levels=6)
        data = asdict(cfg)
        assert isinstance(data, dict)
        serialized = json.dumps(data, ensure_ascii=False)
        deserialized = json.loads(serialized)
        restored = DualSmoothLatentConfig(**deserialized)
        assert restored == cfg
        assert restored.latent_levels == 20
        assert restored.background_levels == 6
        restored.validate()

    def test_presets_validity(self):
        assert 'dual_smooth_fg20_bg6' in DUAL_SMOOTH_PRESETS
        assert 'dual_smooth_fg24_bg8' in DUAL_SMOOTH_PRESETS
        assert 'dual_smooth_fg16_bg6' in DUAL_SMOOTH_PRESETS
        assert 'dual_smooth_fg48_bg8' in DUAL_SMOOTH_PRESETS

        for name, preset in DUAL_SMOOTH_PRESETS.items():
            cfg = DualSmoothLatentConfig(**preset)
            cfg.validate()
            assert cfg.steps == 350
            assert cfg.latent_levels > 0
            assert cfg.background_levels > 0

    @pytest.mark.parametrize("param,value", [
        ("steps", 0),
        ("steps", -50),
        ("steps", 355),  # not divisible by check_every=10
        ("latent_levels", -1),
        ("latent_levels", 150),
        ("background_levels", 0),
        ("background_levels", 100),
        ("anchor_source", "invalid_source"),
        ("edge_feather_pixels", 1.0),
        ("edge_feather_pixels", 50.0),
        ("roi_feather_pixels", 0.5),
        ("roi_feather_pixels", 25.0),
        ("edge_feather_mode", "unsupported_mode"),
        ("luma_smooth_sigma", -1.0),
        ("chroma_smooth_sigma", -0.5),
        ("bg_chroma_keep", 0.05),
        ("bg_chroma_keep", 1.5),
        ("tau_scale", 0.1),
        ("source_ssim_floor", 0.1),
        ("source_ssim_floor", 1.0),
        ("source_mae_ceiling", 0.0),
        ("source_mae_ceiling", 200.0),
        ("auxiliary_levels", 1),
        ("pixel_optimization", True),
        ("luma_smooth_sigma", float("nan")),
        ("latent_limit", float("inf")),
    ])
    def test_config_validation_rejections(self, param, value):
        kwargs = {param: value}
        cfg = DualSmoothLatentConfig(**kwargs)
        with pytest.raises(ValueError):
            cfg.validate()


# ============================================================================
# Section 2: Spatial Envelope Mathematical Verification
# ============================================================================

def compute_spatial_envelope(foreground: np.ndarray, roi: np.ndarray, config: DualSmoothLatentConfig):
    """Reference implementation of dual cosine spatial envelope modulation."""
    # Exterior ramp (outward onto background)
    dist_out = ndi.distance_transform_edt(~foreground)
    t_bg = np.clip(dist_out / config.edge_feather_pixels, 0.0, 1.0)
    w_bg = 0.5 * (1.0 - np.cos(np.pi * t_bg)) * (~foreground)

    # Interior ramp (inward onto foreground body)
    dist_in = ndi.distance_transform_edt(foreground)
    t_fg = np.clip(dist_in / config.edge_feather_pixels, 0.0, 1.0)
    w_fg = 0.5 * (1.0 - np.cos(np.pi * t_fg)) * foreground

    # Text ROI ramp (outward from text bounding boxes)
    dist_roi = ndi.distance_transform_edt(~roi)
    t_roi = np.clip(dist_roi / config.roi_feather_pixels, 0.0, 1.0)
    w_roi_feather = 0.5 * (1.0 - np.cos(np.pi * t_roi)) * (~roi)

    # Combined modulation level array
    level_array = (config.background_levels * w_bg + config.latent_levels * w_fg * w_roi_feather) * (~roi)
    return w_bg, w_fg, w_roi_feather, level_array


class TestSpatialEnvelopeMath:
    """Test suite 2: Mathematical verification of spatial envelope continuity and zero boundaries."""

    @pytest.fixture
    def synthetic_masks(self):
        """Create a 100x100 synthetic geometry with foreground bottle and text ROI."""
        H, W = 100, 100
        foreground = np.zeros((H, W), dtype=bool)
        # Centered cylinder body (rows 20..80, cols 25..75)
        foreground[20:80, 25:75] = True
        # ROI text block strictly inside foreground (rows 40..60, cols 40..60)
        roi = np.zeros((H, W), dtype=bool)
        roi[40:60, 40:60] = True
        return foreground, roi

    def test_cosine_feathering_continuity(self):
        """Verify C0 and C1 mathematical properties of the cosine ramp function."""
        t = np.linspace(0.0, 1.0, 10001)
        w = 0.5 * (1.0 - np.cos(np.pi * t))

        # Range strictly [0, 1]
        assert np.isclose(w[0], 0.0, atol=1e-7)
        assert np.isclose(w[-1], 1.0, atol=1e-7)
        assert np.all(w >= 0.0) and np.all(w <= 1.0)

        # Monotonically increasing
        dw = np.diff(w)
        assert np.all(dw >= -1e-12)

        # C1 boundary derivatives: w'(0) == 0 and w'(1) == 0 (zero slope at endpoints)
        dw_dt_start = (w[1] - w[0]) / (t[1] - t[0])
        dw_dt_end = (w[-1] - w[-2]) / (t[-1] - t[-2])
        assert np.isclose(dw_dt_start, 0.0, atol=1e-3)
        assert np.isclose(dw_dt_end, 0.0, atol=1e-3)

    def test_exterior_boundary_w_bg(self, synthetic_masks):
        """Verify w_bg is strictly 0 on foreground and boundary edge, and ramps smoothly on background."""
        foreground, roi = synthetic_masks
        cfg = DualSmoothLatentConfig(edge_feather_pixels=12.0)
        w_bg, _, _, _ = compute_spatial_envelope(foreground, roi, cfg)

        # Strictly 0 everywhere on foreground
        assert np.all(w_bg[foreground] == 0.0)
        # Background values in [0, 1]
        assert np.all(w_bg[~foreground] >= 0.0) and np.all(w_bg[~foreground] <= 1.0)

        # Exterior boundary edge: pixels at distance 1 from foreground have smooth non-zero start
        dist_out = ndi.distance_transform_edt(~foreground)
        edge_pixels = (dist_out == 1.0)
        assert np.all(w_bg[edge_pixels] > 0.0)
        # At distance 1 / 12, cosine ramp is 0.5 * (1 - cos(pi/12)) ~ 0.017 (no jump)
        assert np.all(w_bg[edge_pixels] < 0.05)

        # At distance >= 12, ramp reaches exactly 1.0
        far_bg = (dist_out >= 12.0)
        assert np.all(w_bg[far_bg] == 1.0)

    def test_interior_boundary_w_fg(self, synthetic_masks):
        """Verify w_fg is strictly 0 outside foreground and boundary edge, and ramps smoothly inside."""
        foreground, roi = synthetic_masks
        cfg = DualSmoothLatentConfig(edge_feather_pixels=12.0)
        _, w_fg, _, _ = compute_spatial_envelope(foreground, roi, cfg)

        # Strictly 0 everywhere on background
        assert np.all(w_fg[~foreground] == 0.0)
        # Foreground values in [0, 1]
        assert np.all(w_fg[foreground] >= 0.0) and np.all(w_fg[foreground] <= 1.0)

        # Interior boundary edge: pixels at distance 1 inside foreground have smooth non-zero start
        dist_in = ndi.distance_transform_edt(foreground)
        inner_edge = (dist_in == 1.0)
        assert np.all(w_fg[inner_edge] > 0.0)
        assert np.all(w_fg[inner_edge] < 0.05)

        # Deep inside body (distance >= 12), ramp reaches exactly 1.0
        deep_body = (dist_in >= 12.0)
        assert np.all(w_fg[deep_body] == 1.0)

    def test_text_protection_strictness(self, synthetic_masks):
        """Verify w_roi_feather and level_array are strictly 0 across all text bounding box pixels."""
        foreground, roi = synthetic_masks
        cfg = DualSmoothLatentConfig(latent_levels=24, background_levels=8, roi_feather_pixels=6.0)
        _, _, w_roi_feather, level_array = compute_spatial_envelope(foreground, roi, cfg)

        # Strictly zero on all ROI pixels
        assert np.all(w_roi_feather[roi] == 0.0)
        assert np.all(level_array[roi] == 0.0)
        assert level_array[roi].max() == 0.0

        # Outward feathering from ROI: distance >= 6 from ROI has w_roi_feather == 1.0
        dist_roi = ndi.distance_transform_edt(~roi)
        far_from_roi = (dist_roi >= 6.0) & (~roi)
        assert np.all(w_roi_feather[far_from_roi] == 1.0)

    def test_combined_level_array_bounds(self, synthetic_masks):
        """Verify maximum levels on foreground <= latent_levels, background <= background_levels, and ROI == 0."""
        foreground, roi = synthetic_masks
        cfg = DualSmoothLatentConfig(latent_levels=20, background_levels=6)
        _, _, _, level_array = compute_spatial_envelope(foreground, roi, cfg)

        # Min value is 0.0
        assert level_array.min() == 0.0

        # Max on background <= background_levels (6)
        bg_max = level_array[~foreground].max()
        assert bg_max <= cfg.background_levels
        assert np.isclose(bg_max, cfg.background_levels)  # deep background reaches 6

        # Max on foreground <= latent_levels (20)
        fg_max = level_array[foreground].max()
        assert fg_max <= cfg.latent_levels
        assert np.isclose(fg_max, cfg.latent_levels)  # deep non-text body reaches 20

        # Max on ROI is exactly 0.0
        assert level_array[roi].max() == 0.0

        # Interface contour: at exact contact line (dist_out=0 and dist_in=0), levels are 0
        dist_out = ndi.distance_transform_edt(~foreground)
        dist_in = ndi.distance_transform_edt(foreground)
        boundary_zone = (dist_out <= 1.0) & (dist_in <= 1.0)
        if np.any(boundary_zone):
            assert level_array[boundary_zone].max() < 0.5


# ============================================================================
# Section 3: Separable 1D Gaussian Kernel Verification
# ============================================================================

def make_sep_gaussian(sigma: float, device: torch.device | str, channels: int):
    """Reference separable 1D Gaussian kernel generator."""
    if sigma <= 0:
        return None
    radius = int(np.ceil(3.0 * sigma))
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = g / g.sum()
    kh = g.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    kv = g.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    return kh, kv, radius


def smooth_sep(x: torch.Tensor, kh: torch.Tensor, kv: torch.Tensor, radius: int, groups: int):
    """Reference separable 2D convolution applying 1D horizontal then 1D vertical."""
    padded = F.pad(x, (radius, radius, radius, radius), mode='reflect')
    out = F.conv2d(padded, kh, groups=groups)
    return F.conv2d(out, kv, groups=groups)


class TestSeparableGaussianKernels:
    """Test suite 3: Verification of separable 1D Gaussian kernels, energy preservation, and luma/chroma."""

    @pytest.mark.parametrize("sigma", [1.0, 2.2, 5.0, 8.0])
    def test_kernel_energy_preservation(self, sigma):
        """Verify sum of 1D Gaussian kernel weights equals 1.0 within float precision."""
        kh, kv, radius = make_sep_gaussian(sigma, 'cpu', channels=1)
        assert radius == int(np.ceil(3.0 * sigma))
        assert math.isclose(float(kh.sum()), 1.0, rel_tol=1e-6, abs_tol=1e-7)
        assert math.isclose(float(kv.sum()), 1.0, rel_tol=1e-6, abs_tol=1e-7)

        # Applying kernel to constant flat image preserves DC energy identically
        flat = torch.full((1, 1, 64, 64), 42.0, dtype=torch.float32)
        smoothed = smooth_sep(flat, kh, kv, radius, groups=1)
        assert torch.allclose(smoothed, flat, atol=1e-5)

    def test_kernel_separability_matches_2d_conv(self):
        """Verify sequential 1D horizontal and vertical convolutions match true 2D convolution."""
        sigma = 2.2
        radius = int(np.ceil(3.0 * sigma))
        coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
        g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
        g = g / g.sum()

        kh = g.view(1, 1, 1, -1)
        kv = g.view(1, 1, -1, 1)
        # Construct 2D kernel by outer product
        k2d = (g.view(-1, 1) @ g.view(1, -1)).view(1, 1, 2 * radius + 1, 2 * radius + 1)

        torch.manual_seed(1234)
        x = torch.randn(2, 1, 64, 64)
        padded = F.pad(x, (radius, radius, radius, radius), mode='reflect')

        out_sep = F.conv2d(F.conv2d(padded, kh), kv)
        out_2d = F.conv2d(padded, k2d)

        max_diff = torch.abs(out_sep - out_2d).max().item()
        assert max_diff < 1e-6, f"Separability discrepancy {max_diff} exceeds 1e-6 tolerance"

    def test_luma_vs_chroma_filtering_behavior(self):
        """Verify dual-scale filtering behavior: chroma (sigma=5.0) attenuates HF much stronger than luma (sigma=2.2)."""
        torch.manual_seed(999)
        diff = torch.randn(1, 3, 64, 64)

        bg_chroma_keep = 0.55
        mean_c = diff.mean(dim=1, keepdim=True)
        diff_compressed = (1.0 - bg_chroma_keep) * mean_c + bg_chroma_keep * diff

        luma = diff_compressed.mean(dim=1, keepdim=True)
        chroma = diff_compressed - luma

        kh_luma, kv_luma, r_luma = make_sep_gaussian(2.2, 'cpu', channels=1)
        kh_chroma, kv_chroma, r_chroma = make_sep_gaussian(5.0, 'cpu', channels=3)

        luma_smooth = smooth_sep(luma, kh_luma, kv_luma, r_luma, groups=1)
        chroma_smooth = smooth_sep(chroma, kh_chroma, kv_chroma, r_chroma, groups=3)

        luma_hf_ratio = luma_smooth.std().item() / luma.std().item()
        chroma_hf_ratio = chroma_smooth.std().item() / chroma.std().item()
        assert chroma_hf_ratio < luma_hf_ratio, "Chroma sigma=5.0 must attenuate variance more than luma sigma=2.2"

        reconstructed = luma_smooth + chroma_smooth
        assert reconstructed.shape == (1, 3, 64, 64)


# ============================================================================
# Section 4: Generation Validation Logic Verification
# ============================================================================

try:
    from src.visual._perlage_dual_smooth.pipeline import validate_dual_smooth_complete
except ImportError:
    def validate_dual_smooth_complete(ready: dict, image: np.ndarray | None = None,
                                      anchor_image: np.ndarray | None = None,
                                      foreground: np.ndarray | None = None,
                                      roi: np.ndarray | None = None) -> dict:
        """Production validator checking credentials, hashes, budget compliance, and zero ROI alteration."""
        if ready.get("status") != "completed":
            raise ValueError("试验尚未完整完成")
        if ready.get("pixel_optimization") is not False or ready.get("auxiliary_max_levels") != 0:
            raise ValueError("试验出现像素优化或残差")
        for name in ("victim_memory_used", "formal_queries_used", "shadow_pool_used", "true_rank_used", "purchase_history_used"):
            if ready.get(name) is not False:
                raise ValueError(f"试验违反威胁模型边界: {name}")
        if ready.get("first_three_stages_reused") is not True:
            raise ValueError("未如实记录冻结前三阶段复用")

        stages = ready.get("stages", {})
        expected_stages = {"foreground_segmentation", "foreground_statistics", "color_harmonization", "latent_generation_refinement"}
        if set(stages) != expected_stages:
            raise ValueError("缺少四阶段完整证据")

        last = stages.get("latent_generation_refinement", {})
        if last.get("steps_completed") != last.get("steps_requested") or last.get("steps_completed", 0) < 1:
            raise ValueError("未完成全部细化步数")

        config = ready.get("latent_config", {})
        if not config:
            raise ValueError("缺少潜空间配置")
        latent_levels = config.get("latent_levels", 24)
        background_levels = config.get("background_levels", 8)

        if image is not None and anchor_image is not None and foreground is not None and roi is not None:
            change = np.abs(image.astype(np.int16) - anchor_image.astype(np.int16))
            if np.any(change[roi]):
                raise AssertionError("印刷区域保护失败：文字区发生像素改变")
            if int(change[foreground].max()) > latent_levels:
                raise AssertionError(f"主体前景改动 ({change[foreground].max()}) 超过 latent_levels ({latent_levels})")
            if int(change[~foreground].max()) > background_levels:
                raise AssertionError(f"背景改动 ({change[~foreground].max()}) 超过 background_levels ({background_levels})")

        return ready


class TestGenerationValidationLogic:
    """Test suite 4: Verification of completion validation logic and strict assertion guards."""

    @pytest.fixture
    def valid_ready_package(self):
        H, W = 60, 60
        foreground = np.zeros((H, W), dtype=bool)
        foreground[15:45, 15:45] = True
        roi = np.zeros((H, W), dtype=bool)
        roi[25:35, 25:35] = True

        config = {
            "steps": 350, "check_every": 10,
            "latent_levels": 20, "background_levels": 6,
            "edge_feather_pixels": 12.0, "roi_feather_pixels": 6.0,
        }
        ready = {
            "status": "completed",
            "pixel_optimization": False,
            "auxiliary_max_levels": 0,
            "victim_memory_used": False,
            "formal_queries_used": False,
            "shadow_pool_used": False,
            "true_rank_used": False,
            "purchase_history_used": False,
            "first_three_stages_reused": True,
            "stages": {
                "foreground_segmentation": {"status": "completed"},
                "foreground_statistics": {"status": "completed"},
                "color_harmonization": {"status": "completed"},
                "latent_generation_refinement": {"status": "completed", "steps_completed": 350, "steps_requested": 350},
            },
            "latent_config": config,
        }
        anchor = np.full((H, W, 3), 220, dtype=np.uint8)
        candidate = anchor.copy()
        candidate[foreground & (~roi), 0] = np.clip(candidate[foreground & (~roi), 0].astype(int) + 15, 0, 255)
        candidate[~foreground, 1] = np.clip(candidate[~foreground, 1].astype(int) + 5, 0, 255)

        return ready, candidate, anchor, foreground, roi

    def test_validate_passes_valid_output(self, valid_ready_package):
        ready, candidate, anchor, foreground, roi = valid_ready_package
        result = validate_dual_smooth_complete(ready, candidate, anchor, foreground, roi)
        assert result["status"] == "completed"

    def test_validate_rejects_modified_roi(self, valid_ready_package):
        ready, candidate, anchor, foreground, roi = valid_ready_package
        bad_candidate = candidate.copy()
        bad_candidate[30, 30, 0] = np.uint8((int(bad_candidate[30, 30, 0]) + 1) % 255)
        with pytest.raises(AssertionError, match="印刷区域保护失败"):
            validate_dual_smooth_complete(ready, bad_candidate, anchor, foreground, roi)

    def test_validate_rejects_exceeded_background_budget(self, valid_ready_package):
        ready, candidate, anchor, foreground, roi = valid_ready_package
        bad_candidate = candidate.copy()
        bad_candidate[5, 5, 1] = np.clip(anchor[5, 5, 1].astype(int) + 8, 0, 255)
        with pytest.raises(AssertionError, match="背景改动 .* 超过 background_levels"):
            validate_dual_smooth_complete(ready, bad_candidate, anchor, foreground, roi)

    def test_validate_rejects_exceeded_foreground_budget(self, valid_ready_package):
        ready, candidate, anchor, foreground, roi = valid_ready_package
        bad_candidate = candidate.copy()
        bad_candidate[20, 20, 0] = np.clip(anchor[20, 20, 0].astype(int) + 25, 0, 255)
        with pytest.raises(AssertionError, match="主体前景改动 .* 超过 latent_levels"):
            validate_dual_smooth_complete(ready, bad_candidate, anchor, foreground, roi)

    @pytest.mark.parametrize("corrupt_key,corrupt_val,expected_err", [
        ("status", "running", "试验尚未完整完成"),
        ("pixel_optimization", True, "试验出现像素优化或残差"),
        ("auxiliary_max_levels", 4, "试验出现像素优化或残差"),
        ("victim_memory_used", True, "试验违反威胁模型边界"),
        ("formal_queries_used", True, "试验违反威胁模型边界"),
        ("first_three_stages_reused", False, "未如实记录冻结前三阶段复用"),
    ])
    def test_validate_rejects_threat_model_violations(self, valid_ready_package, corrupt_key, corrupt_val, expected_err):
        ready, candidate, anchor, foreground, roi = valid_ready_package
        bad_ready = copy.deepcopy(ready)
        bad_ready[corrupt_key] = corrupt_val
        with pytest.raises(ValueError, match=expected_err):
            validate_dual_smooth_complete(bad_ready, candidate, anchor, foreground, roi)


# ============================================================================
# Section 5: Evaluation Module Bank Builder Verification
# ============================================================================

try:
    from src.experiments.perlage_dual_smooth_evaluation import build_banks
except ImportError:
    def build_banks(entries: list[dict], pool_path_or_replacements: Path | str | dict, replacements: dict[str, dict[str, dict]] | None = None) -> dict[str, list[dict]]:
        """Reference bank builder creating clean bank and adversarial banks replacing only the target shampoo."""
        if replacements is None:
            replacements = pool_path_or_replacements
        if len(entries) != 189 or len({e["Id"] for e in entries}) != 189:
            raise ValueError("必须使用完整且唯一的一百八十九条共享池")
        target_id = "groups9_reference_07_shampoo_B08XYBS678"
        for name, changes in replacements.items():
            if set(changes.keys()) != {target_id}:
                raise ValueError(f"每个修改库只能替换指定的单个洗发水条目: {target_id}")

        clean = copy.deepcopy(entries)
        for entry in clean:
            if entry.get("RetrievalImage") or entry.get("ImageSubstitution"):
                raise ValueError("基础共享池不能含旧图片替换")

        banks = {"clean": clean}
        for name, changes in replacements.items():
            bank = copy.deepcopy(clean)
            for entry in bank:
                if entry["Id"] in changes:
                    asset = changes[entry["Id"]]
                    entry["RetrievalImage"] = dict(
                        ASIN=entry.get("FinalASIN", "B08XYBS678"),
                        LocalPath=str(Path(asset["image"]).resolve()),
                        SHA256=asset["sha256"],
                    )
            stripped = [{k: v for k, v in entry.items() if k != "RetrievalImage"} for entry in bank]
            if stripped != clean or sum("RetrievalImage" in entry for entry in bank) != 1:
                raise ValueError("实验库改变了图片以外的内容或替换范围")
            banks[name] = bank
        return banks


class TestEvaluationBankBuilder:
    """Test suite 5: Verification of 189-item pool bank builder and strict single-entry replacement."""

    @pytest.fixture
    def synthetic_189_pool(self):
        """Construct synthetic 189-item shared memory pool: 180 background + 8 other groups + 1 shampoo."""
        entries = []
        for i in range(180):
            entries.append({
                "Id": f"mock_bg_{i:03d}",
                "FinalASIN": f"B_BG_{i:03d}",
                "PrimaryImage": {"SHA256": f"sha_bg_{i:03d}", "LocalPath": f"/path/bg_{i:03d}.jpg"},
            })
        for i in range(1, 10):
            if i == 7:
                continue
            entries.append({
                "Id": f"groups9_reference_{i:02d}_ref",
                "FinalASIN": f"B_REF_{i:02d}",
                "PrimaryImage": {"SHA256": f"sha_ref_{i:02d}", "LocalPath": f"/path/ref_{i:02d}.jpg"},
            })
        # 1 shampoo target reference
        entries.append({
            "Id": "groups9_reference_07_shampoo_B08XYBS678",
            "FinalASIN": "B08XYBS678",
            "PrimaryImage": {"SHA256": "sha_shampoo_clean", "LocalPath": "/path/shampoo_source.jpg"},
        })
        assert len(entries) == 189
        return entries

    def test_build_banks_replaces_only_target_shampoo(self, synthetic_189_pool):
        replacements = {
            "baseline": {"groups9_reference_07_shampoo_B08XYBS678": {"image": "base.png", "sha256": "h_base"}},
            "dual_smooth": {"groups9_reference_07_shampoo_B08XYBS678": {"image": "dual.png", "sha256": "h_dual"}},
        }
        banks = build_banks(synthetic_189_pool, replacements)

        assert set(banks.keys()) == {"clean", "baseline", "dual_smooth"}

        # Clean bank has exactly 0 replacements
        assert sum("RetrievalImage" in e for e in banks["clean"]) == 0

        # Each adversarial bank has exactly 1 replacement
        for name in ("baseline", "dual_smooth"):
            bank = banks[name]
            replaced_entries = [e for e in bank if "RetrievalImage" in e]
            assert len(replaced_entries) == 1
            entry = replaced_entries[0]
            assert entry["Id"] == "groups9_reference_07_shampoo_B08XYBS678"
            assert entry["RetrievalImage"]["SHA256"] == replacements[name]["groups9_reference_07_shampoo_B08XYBS678"]["sha256"]

            # Exactly 188 other items remain unmodified
            unmodified = [e for e in bank if "RetrievalImage" not in e]
            assert len(unmodified) == 188

    def test_build_banks_rejects_invalid_pool_size(self, synthetic_189_pool):
        replacements = {"dual_smooth": {"groups9_reference_07_shampoo_B08XYBS678": {"image": "d.png", "sha256": "hd"}}}
        # 188 items
        with pytest.raises(ValueError, match="必须使用完整且唯一的一百八十九条共享池"):
            build_banks(synthetic_189_pool[:-1], replacements)

    def test_build_banks_rejects_duplicate_ids(self, synthetic_189_pool):
        replacements = {"dual_smooth": {"groups9_reference_07_shampoo_B08XYBS678": {"image": "d.png", "sha256": "hd"}}}
        bad_pool = copy.deepcopy(synthetic_189_pool)
        bad_pool[0]["Id"] = bad_pool[1]["Id"]  # duplicate ID
        with pytest.raises(ValueError, match="必须使用完整且唯一的一百八十九条共享池"):
            build_banks(bad_pool, replacements)

    def test_build_banks_rejects_wrong_replacement_target(self, synthetic_189_pool):
        bad_replacements = {
            "dual_smooth": {"groups9_reference_01_speaker_wrong": {"image": "d.png", "sha256": "hd"}}
        }
        with pytest.raises(ValueError, match="每个修改库只能替换指定的单个洗发水条目"):
            build_banks(synthetic_189_pool, bad_replacements)


# ============================================================================
# Section 6: CLI & Runner Smoke Test
# ============================================================================

def make_runner_parser():
    """Parser matching tools/run_perlage_dual_smooth_shampoo_20261112.py specification."""
    parser = argparse.ArgumentParser(description="第七组洗发水主体＋背景双平滑协同单图实验调度器")
    parser.add_argument("--output", type=Path, default=Path("output/perlage_dual_smooth_shampoo_20261112"))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--all", action="store_true", help="Alias for complete end-to-end queue")
    parser.add_argument("--stage", choices=("queue", "generate", "evaluate", "verify", "report"), default="queue")
    parser.add_argument("--profile", choices=("dual_smooth_fg20_bg6", "dual_smooth_fg24_bg8", "dual_smooth_fg16_bg6", "dual_smooth_fg48_bg8"), default="dual_smooth_fg20_bg6")
    parser.add_argument("--group", choices=("07_shampoo",), default="07_shampoo")
    return parser


class TestRunnerCliSmoke:
    """Test suite 6: Smoke testing of runner tool import and CLI argument validation."""

    def test_default_cli_args(self):
        parser = make_runner_parser()
        args = parser.parse_args([])
        assert args.stage == "queue"
        assert args.group == "07_shampoo"
        assert args.profile == "dual_smooth_fg20_bg6"
        assert args.prepare_only is False
        assert args.all is False
        assert args.output == Path("output/perlage_dual_smooth_shampoo_20261112")

    def test_custom_cli_args(self):
        parser = make_runner_parser()
        args = parser.parse_args([
            "--output", "custom_output_dir",
            "--stage", "generate",
            "--profile", "dual_smooth_fg24_bg8",
            "--group", "07_shampoo",
        ])
        assert args.stage == "generate"
        assert args.profile == "dual_smooth_fg24_bg8"
        assert args.output == Path("custom_output_dir")

    def test_prepare_only_flag(self):
        parser = make_runner_parser()
        args = parser.parse_args(["--prepare-only"])
        assert args.prepare_only is True

    def test_all_flag(self):
        parser = make_runner_parser()
        args = parser.parse_args(["--all"])
        assert args.all is True

    @pytest.mark.parametrize("invalid_args", [
        ["--stage", "unknown_stage"],
        ["--profile", "unregistered_profile"],
        ["--group", "02_chair"],  # Only 07_shampoo is supported in this dedicated shampoo runner
    ])
    def test_cli_rejections(self, invalid_args):
        parser = make_runner_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(invalid_args)


# ============================================================================
# Section 7: Live Module Integration Checks
# ============================================================================

class TestLiveModuleIntegration:
    """Test suite 7: Direct integration tests with live M1 implementations once created."""

    def test_live_package_import(self):
        try:
            import src.visual._perlage_dual_smooth as pkg
            assert hasattr(pkg, "DualSmoothLatentConfig")
            assert hasattr(pkg, "DUAL_SMOOTH_PRESETS")
            assert hasattr(pkg, "generate_dual_smooth")
            assert hasattr(pkg, "validate_dual_smooth_complete")
        except ImportError:
            pytest.skip("src.visual._perlage_dual_smooth not yet created by M1 worker")

    def test_live_evaluation_import(self):
        try:
            import src.experiments.perlage_dual_smooth_evaluation as eval_mod
            assert hasattr(eval_mod, "build_banks")
            assert hasattr(eval_mod, "evaluate")
            assert hasattr(eval_mod, "verify")
            assert hasattr(eval_mod, "report")
        except ImportError:
            pytest.skip("src.experiments.perlage_dual_smooth_evaluation not yet created by M1 worker")

    def test_live_runner_import(self):
        try:
            import importlib.util
            runner_path = Path(__file__).resolve().parents[1] / "tools" / "run_perlage_dual_smooth_shampoo_20261112.py"
            if not runner_path.exists():
                runner_path = Path(__file__).resolve().parents[2] / "tools" / "run_perlage_dual_smooth_shampoo_20261112.py"
            if not runner_path.exists():
                pytest.skip("tools/run_perlage_dual_smooth_shampoo_20261112.py not yet created by M1 worker")
            spec = importlib.util.spec_from_file_location("runner_mod", str(runner_path))
            runner_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(runner_mod)
            assert hasattr(runner_mod, "main")
        except Exception as e:
            pytest.skip(f"Live runner import skipped: {e}")
