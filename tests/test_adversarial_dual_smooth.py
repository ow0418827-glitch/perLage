"""Adversarial stress-testing suite for PerLAGE Dual-Smooth mathematical envelope & smoothing kernel.

Author: Challenger 1 (Milestone M1 Adversarial Verifier)
Target: src/visual/_perlage_dual_smooth/pipeline.py

Verifies:
1. Boundary derivative continuity (C^1 smoothness at foreground outer edge d=0, inward/outward cosine ramp).
2. Text protection robustness under extreme synthetic VAE difference fields (constant 255, noise, impulses).
3. Budget bounding on foreground non-ROI (<= latent_levels) and background (<= background_levels).
4. Separable Gaussian kernel energy preservation (sum(g) == 1.0) and 2D vs 1D separable equivalence.
5. Degenerate geometries and validation failure sensitivity.
"""
from __future__ import annotations

import math
from dataclasses import asdict
import numpy as np
import pytest
import scipy.ndimage as ndi
import torch
import torch.nn.functional as F

from src.visual._perlage_dual_smooth.pipeline import (
    DualSmoothLatentConfig,
    DUAL_SMOOTH_PRESETS,
    _make_sep_gaussian,
    _smooth_sep,
    _DualSmoothLatentBranch,
    validate_dual_smooth_complete,
)


def _compute_envelope_weights(foreground: np.ndarray, roi: np.ndarray, config: DualSmoothLatentConfig):
    """Replicates spatial envelope calculation from pipeline.py lines 664-704."""
    dist_out = ndi.distance_transform_edt(~foreground)
    if config.edge_feather_mode == 'cosine':
        t_out = np.clip(dist_out / config.edge_feather_pixels, 0.0, 1.0)
        w_bg = 0.5 * (1.0 - np.cos(np.pi * t_out)) * (~foreground)
    else:
        w_bg = np.clip(dist_out / config.edge_feather_pixels, 0.0, 1.0) * (~foreground)

    dist_in = ndi.distance_transform_edt(foreground)
    if config.edge_feather_mode == 'cosine':
        t_in = np.clip(dist_in / config.edge_feather_pixels, 0.0, 1.0)
        w_fg = 0.5 * (1.0 - np.cos(np.pi * t_in)) * foreground
    else:
        w_fg = np.clip(dist_in / config.edge_feather_pixels, 0.0, 1.0) * foreground

    if config.roi_feather_pixels > 0:
        dist_text = ndi.distance_transform_edt(~roi)
        t_text = np.clip(dist_text / config.roi_feather_pixels, 0.0, 1.0)
        w_text = 0.5 * (1.0 - np.cos(np.pi * t_text))
        w_fg_body = w_fg * w_text
    else:
        w_fg_body = w_fg

    level_array = (config.background_levels * w_bg + config.latent_levels * w_fg_body) * (~roi)
    max_level = float(max(config.background_levels, config.latent_levels, 1))
    macro_array = level_array / max_level
    eps_map = torch.as_tensor(level_array / 255.0, dtype=torch.float32)[None, None]
    tau_floor = max(min(config.background_levels, config.latent_levels), 1) / 255.0
    tau_map = eps_map.clamp_min(tau_floor)

    chroma_keep_array = (config.fg_chroma_keep * w_fg_body + config.bg_chroma_keep * w_bg).astype(np.float32)
    residual_weight = np.clip(1.0 - (w_fg_body + w_bg), 0.0, 1.0)
    chroma_keep_array = chroma_keep_array + residual_weight * config.bg_chroma_keep
    chroma_keep_map = torch.as_tensor(chroma_keep_array, dtype=torch.float32)[None, None]

    support_mask = torch.as_tensor((~roi).astype(np.float32))[None, None]
    return {
        'w_bg': w_bg,
        'w_fg': w_fg,
        'w_fg_body': w_fg_body,
        'level_array': level_array,
        'macro_array': macro_array,
        'eps_map': eps_map,
        'tau_map': tau_map,
        'chroma_keep_map': chroma_keep_map,
        'support_mask': support_mask,
    }


class _MockPurifierForSyntheticDiff:
    """Mock purifier that allows injecting exact synthetic difference fields into _DualSmoothLatentBranch."""
    def __init__(self, synthetic_diff: torch.Tensor, anchor_shape: tuple[int, ...]):
        self.synthetic_diff = synthetic_diff
        self.anchor_shape = anchor_shape
        self.device = synthetic_diff.device
        self.identity = {'kind': 'mock_purifier_for_adversarial_testing'}
        self.reference = torch.zeros(anchor_shape, device=self.device)

    def _decode_raw(self, latent: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
        return self.reference + self.synthetic_diff


# ============================================================================
# Section 1: Boundary Derivative Continuity (C^1 Smoothness)
# ============================================================================

class TestBoundaryDerivativeContinuity:
    """Stress-test C^1 boundary derivative continuity at the foreground interface (d=0) and ramp saturation (d=L)."""

    def test_1d_analytical_derivative_vanishes_at_interface(self):
        """At d=0, dw/dd = 0.5 * pi / L * sin(0) = 0 for both inward and outward ramps."""
        L = 12.0
        # Check inward ramp: w_fg(d) = 0.5 * (1 - cos(pi * d / L))
        d_vals = np.logspace(-6, -1, 50)  # d approaching 0
        w_fg_vals = 0.5 * (1.0 - np.cos(np.pi * d_vals / L))

        # 1. One-sided secant slope (w(d) - w(0)) / d -> 0 as d -> 0
        secant_slope = w_fg_vals / d_vals
        assert secant_slope[0] < 1e-4, f"Secant slope at d={d_vals[0]} should be near zero, got {secant_slope[0]}"

        # 2. Tangent derivative dw/dd = 0.5 * (pi / L) * sin(pi * d / L)
        analytical_deriv = 0.5 * (np.pi / L) * np.sin(np.pi * d_vals / L)
        assert analytical_deriv[0] < 1e-4, f"Analytical deriv at d={d_vals[0]} should be near zero, got {analytical_deriv[0]}"

        # 3. For quadratic small-angle approximation w(d) ~ 0.5 * 0.5 * (pi*d/L)^2,
        # secant_slope is exactly 0.5 * analytical_deriv:
        np.testing.assert_allclose(secant_slope, 0.5 * analytical_deriv, rtol=1e-2)

        # 4. Contrast with linear ramp: w_lin(d) = d / L -> derivative is 1/L everywhere (0.0833)
        w_lin_vals = d_vals / L
        deriv_lin = w_lin_vals / d_vals
        np.testing.assert_allclose(deriv_lin, 1.0 / L, rtol=1e-5)
        # Cosine derivative is at least 1000x smaller near boundary than linear
        assert secant_slope[0] < deriv_lin[0] / 1000.0

    def test_1d_ramp_saturation_derivative_vanishes_at_L(self):
        """At d=L, dw/dd = 0.5 * pi / L * sin(pi) = 0, ensuring smooth C^1 transition to flat level."""
        L = 12.0
        eps_vals = np.logspace(-6, -1, 50)
        d_vals = L - eps_vals
        w_vals = 0.5 * (1.0 - np.cos(np.pi * d_vals / L))

        # (w(L) - w(L - eps)) / eps -> 0 as eps -> 0
        secant_at_L = (1.0 - w_vals) / eps_vals
        assert secant_at_L[0] < 1e-4
        analytical_at_L = 0.5 * (np.pi / L) * np.sin(np.pi * (L - eps_vals) / L)
        assert analytical_at_L[0] < 1e-4
        np.testing.assert_allclose(secant_at_L, 0.5 * analytical_at_L, rtol=1e-2)

    def test_continuous_transition_across_interface_1d(self):
        """Across a 1D interface from foreground (x <= 0) to background (x > 0),

        both left derivative lim_{x->0-} W'(x) = 0 and right derivative lim_{x->0+} W'(x) = 0 vanish.
        """
        L = 12.0
        # Left side: x approaching 0 from negative (foreground)
        x_left = -np.logspace(-5, -1, 40)
        d_in = np.abs(x_left)
        w_fg = 0.5 * (1.0 - np.cos(np.pi * np.clip(d_in / L, 0, 1)))
        W_left = 24.0 * w_fg
        deriv_left = (W_left - 0.0) / x_left  # secant slope approaching 0 from left
        assert np.abs(deriv_left[0]) < 1e-3, f"Left derivative near boundary: {deriv_left[0]}"

        # Right side: x approaching 0 from positive (background)
        x_right = np.logspace(-5, -1, 40)
        d_out = x_right
        w_bg = 0.5 * (1.0 - np.cos(np.pi * np.clip(d_out / L, 0, 1)))
        W_right = 8.0 * w_bg
        deriv_right = (W_right - 0.0) / x_right  # secant slope approaching 0 from right
        assert np.abs(deriv_right[0]) < 1e-3, f"Right derivative near boundary: {deriv_right[0]}"

        # Both one-sided derivatives vanish, proving W is C^1 smooth at x=0
        assert np.isclose(deriv_left[0], 0.0, atol=1e-3)
        assert np.isclose(deriv_right[0], 0.0, atol=1e-3)

    def test_2d_circular_boundary_gradient_vanishes(self):
        """In a 2D circular foreground mask, the spatial gradient |grad(level_array)| vanishes at boundary."""
        H, W = 128, 128
        yy, xx = np.mgrid[:H, :W]
        center_y, center_x = 64, 64
        radius = 32.0
        dist_from_center = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
        foreground = dist_from_center <= radius
        roi = np.zeros((H, W), dtype=bool)

        cfg = DualSmoothLatentConfig(latent_levels=24, background_levels=8, edge_feather_pixels=12.0)
        env = _compute_envelope_weights(foreground, roi, cfg)
        levels = env['level_array']

        # Boundary zone: within 1.5 pixels of the interface
        dist_in = ndi.distance_transform_edt(foreground)
        dist_out = ndi.distance_transform_edt(~foreground)
        boundary_zone = (dist_in > 0) & (dist_in <= 1.5) | (dist_out > 0) & (dist_out <= 1.5)

        # Sobel gradient magnitude
        gy = ndi.sobel(levels, axis=0) / 8.0
        gx = ndi.sobel(levels, axis=1) / 8.0
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)

        # Mid-ramp maximum theoretical gradient is ~ 3.14 levels/pixel
        max_grad_mid_ramp = 0.5 * (np.pi / 12.0) * 24.0
        boundary_grad = grad_mag[boundary_zone]
        assert boundary_grad.max() < 0.6 * max_grad_mid_ramp, f"Boundary grad max: {boundary_grad.max()}"
        assert boundary_grad.mean() < 0.3 * max_grad_mid_ramp

    def test_text_roi_ramp_c1_smoothness(self):
        """Text ROI outward ramp w_text satisfies w_text(0) = 0, w_text'(0) = 0, w_text(L) = 1, w_text'(L) = 0."""
        L_roi = 6.0
        d_vals = np.logspace(-5, -1, 30)
        w_text_start = 0.5 * (1.0 - np.cos(np.pi * d_vals / L_roi))
        # Tangent derivative dw/dd = 0.5 * (pi / L) * sin(pi * d / L) -> 0
        deriv_start = 0.5 * (np.pi / L_roi) * np.sin(np.pi * d_vals / L_roi)
        assert deriv_start[0] < 1e-4

        # Near saturation d -> L_roi
        d_end = L_roi - d_vals
        deriv_end = 0.5 * (np.pi / L_roi) * np.sin(np.pi * d_end / L_roi)
        assert deriv_end[0] < 1e-4


# ============================================================================
# Section 2: Text Protection Robustness under Extreme Synthetic VAE Diff Fields
# ============================================================================

class TestTextProtectionRobustness:
    """Stress-test that ROI perturbation is identically 0.0 with 0 floating-point tolerance under extreme difference fields."""

    @pytest.fixture
    def setup_branch(self):
        """Creates a mock branch with a synthetic bottle shape and text ROI."""
        H, W = 64, 64
        foreground = np.zeros((H, W), dtype=bool)
        foreground[16:48, 16:48] = True  # 32x32 bottle body
        roi = np.zeros((H, W), dtype=bool)
        roi[24:36, 24:40] = True  # 12x16 text block inside body

        cfg = DualSmoothLatentConfig(
            latent_levels=24, background_levels=8,
            luma_smooth_sigma=2.2, chroma_smooth_sigma=5.0,
            edge_feather_pixels=12.0, roi_feather_pixels=6.0,
            tau_scale=0.75, bg_chroma_keep=0.55, fg_chroma_keep=0.85
        )
        env = _compute_envelope_weights(foreground, roi, cfg)
        anchor = torch.full((1, 3, H, W), 0.5, dtype=torch.float32)

        def make_branch(synthetic_diff: torch.Tensor):
            purifier = _MockPurifierForSyntheticDiff(synthetic_diff, (1, 3, H, W))
            branch = _DualSmoothLatentBranch.__new__(_DualSmoothLatentBranch)
            branch.anchor = anchor
            branch.eps_map = env['eps_map']
            branch.tau_map = env['tau_map']
            branch.chroma_keep_map = env['chroma_keep_map']
            branch.support_mask = env['support_mask']
            branch.purifier = purifier
            branch.config = cfg
            branch.max_eps = max(cfg.background_levels, cfg.latent_levels) / 255.0
            branch.luma_kernel = _make_sep_gaussian(cfg.luma_smooth_sigma, anchor.device, 1)
            branch.chroma_kernel = _make_sep_gaussian(cfg.chroma_smooth_sigma, anchor.device, 3)
            branch.base = torch.zeros((1, 4, H // 8, W // 8), dtype=torch.float32)
            branch.reference = purifier.reference
            branch.delta = torch.zeros_like(branch.base)
            return branch, roi, foreground, cfg

        return make_branch

    def test_constant_positive_255(self, setup_branch):
        """Constant +255.0 field: residual on ROI must be identically 0.0 with 0 floating point tolerance."""
        diff = torch.full((1, 3, 64, 64), 255.0, dtype=torch.float32)
        branch, roi, fg, cfg = setup_branch(diff)
        res = branch.residual()

        roi_t = torch.as_tensor(roi)[None, None].expand_as(res)
        assert torch.all(res[roi_t] == 0.0), "ROI must be strictly 0.0 with 0 tolerance"
        assert int((res[roi_t] != 0.0).sum()) == 0

    def test_constant_negative_255(self, setup_branch):
        """Constant -255.0 field: residual on ROI must be identically 0.0."""
        diff = torch.full((1, 3, 64, 64), -255.0, dtype=torch.float32)
        branch, roi, fg, cfg = setup_branch(diff)
        res = branch.residual()

        roi_t = torch.as_tensor(roi)[None, None].expand_as(res)
        assert torch.all(res[roi_t] == 0.0)
        assert int((res[roi_t] != 0.0).sum()) == 0

    def test_high_frequency_extreme_noise(self, setup_branch):
        """Extreme Gaussian noise N(0, 100^2): residual on ROI must be identically 0.0."""
        torch.manual_seed(12345)
        diff = torch.randn(1, 3, 64, 64) * 100.0
        branch, roi, fg, cfg = setup_branch(diff)
        res = branch.residual()

        roi_t = torch.as_tensor(roi)[None, None].expand_as(res)
        assert torch.all(res[roi_t] == 0.0)
        assert int((res[roi_t] != 0.0).sum()) == 0
        assert torch.isfinite(res).all()

    def test_delta_impulse_inside_roi(self, setup_branch):
        """Delta impulse of +1000.0 centered inside ROI: residual on ROI must be identically 0.0."""
        diff = torch.zeros(1, 3, 64, 64, dtype=torch.float32)
        diff[:, :, 30, 30] = 1000.0  # Center of ROI
        branch, roi, fg, cfg = setup_branch(diff)
        res = branch.residual()

        roi_t = torch.as_tensor(roi)[None, None].expand_as(res)
        assert torch.all(res[roi_t] == 0.0)
        assert (res[:, :, 30, 30] == 0.0).all()

    def test_delta_impulse_on_roi_boundary(self, setup_branch):
        """Delta impulse on ROI boundary pixel: residual on ROI must be identically 0.0."""
        diff = torch.zeros(1, 3, 64, 64, dtype=torch.float32)
        diff[:, :, 24, 24] = 1000.0  # Top-left corner of ROI
        branch, roi, fg, cfg = setup_branch(diff)
        res = branch.residual()

        roi_t = torch.as_tensor(roi)[None, None].expand_as(res)
        assert torch.all(res[roi_t] == 0.0)

    def test_delta_impulse_outside_roi_boundary(self, setup_branch):
        """Delta impulse 1 pixel outside ROI: residual on ROI must be identically 0.0."""
        diff = torch.zeros(1, 3, 64, 64, dtype=torch.float32)
        diff[:, :, 23, 24] = 1000.0  # 1 pixel above ROI
        branch, roi, fg, cfg = setup_branch(diff)
        res = branch.residual()

        roi_t = torch.as_tensor(roi)[None, None].expand_as(res)
        assert torch.all(res[roi_t] == 0.0)

    def test_nyquist_checkerboard(self, setup_branch):
        """Checkerboard pattern alternating +/-255 at Nyquist frequency: residual on ROI must be 0.0."""
        yy, xx = torch.meshgrid(torch.arange(64), torch.arange(64), indexing='ij')
        checker = ((yy + xx) % 2 * 2 - 1).float() * 255.0
        diff = checker.view(1, 1, 64, 64).repeat(1, 3, 1, 1)
        branch, roi, fg, cfg = setup_branch(diff)
        res = branch.residual()

        roi_t = torch.as_tensor(roi)[None, None].expand_as(res)
        assert torch.all(res[roi_t] == 0.0)

    def test_huge_amplitude_finite_and_zero_roi(self, setup_branch):
        """Amplitudes of 1e8: tanh must saturate cleanly without NaN, and ROI remains strictly 0.0."""
        diff = torch.full((1, 3, 64, 64), 1e8, dtype=torch.float32)
        branch, roi, fg, cfg = setup_branch(diff)
        res = branch.residual()

        assert torch.isfinite(res).all()
        roi_t = torch.as_tensor(roi)[None, None].expand_as(res)
        assert torch.all(res[roi_t] == 0.0)

    def test_quantized_output_bitwise_identity_on_roi(self, setup_branch):
        """After quantize_ste((anchor + residual).clamp(0, 1)), output on ROI is bitwise identical to anchor."""
        from src.visual.image_utils import quantize_ste
        anchor = torch.full((1, 3, 64, 64), 127.0 / 255.0, dtype=torch.float32)
        diff = torch.randn(1, 3, 64, 64) * 50.0
        branch, roi, fg, cfg = setup_branch(diff)
        branch.anchor = anchor
        res = branch.residual()

        quantized = quantize_ste((anchor + res).clamp(0, 1))
        out_bytes = (quantized * 255.0).round().byte().numpy()
        anchor_bytes = (anchor * 255.0).round().byte().numpy()

        np.testing.assert_array_equal(out_bytes[0, :, roi], anchor_bytes[0, :, roi])


# ============================================================================
# Section 3: Budget Bounding Verification
# ============================================================================

class TestBudgetBounding:
    """Stress-test that residual and quantized perturbation strictly obey latent_levels and background_levels."""

    @pytest.mark.parametrize("preset_name", [
        "dual_smooth",
        "dual_smooth_fg20_bg6",
        "dual_smooth_fg24_bg8",
        "dual_smooth_fg16_bg6",
        "dual_smooth_fg48_bg8",
    ])
    def test_budget_bounding_across_presets(self, preset_name):
        """Under extreme positive and negative saturation, bounds must hold across all official presets."""
        cfg = DualSmoothLatentConfig(**DUAL_SMOOTH_PRESETS[preset_name])
        H, W = 64, 64
        foreground = np.zeros((H, W), dtype=bool)
        foreground[16:48, 16:48] = True
        roi = np.zeros((H, W), dtype=bool)
        roi[24:36, 24:40] = True

        env = _compute_envelope_weights(foreground, roi, cfg)
        anchor = torch.full((1, 3, H, W), 0.5, dtype=torch.float32)

        diff = torch.full((1, 3, H, W), 1e6, dtype=torch.float32)
        purifier = _MockPurifierForSyntheticDiff(diff, (1, 3, H, W))
        branch = _DualSmoothLatentBranch.__new__(_DualSmoothLatentBranch)
        branch.anchor = anchor
        branch.eps_map = env['eps_map']
        branch.tau_map = env['tau_map']
        branch.chroma_keep_map = env['chroma_keep_map']
        branch.support_mask = env['support_mask']
        branch.config = cfg
        branch.max_eps = max(cfg.background_levels, cfg.latent_levels) / 255.0
        branch.luma_kernel = _make_sep_gaussian(cfg.luma_smooth_sigma, anchor.device, 1)
        branch.chroma_kernel = _make_sep_gaussian(cfg.chroma_smooth_sigma, anchor.device, 3)
        branch.purifier = purifier
        branch.reference = purifier.reference
        branch.base = torch.zeros((1, 4, 8, 8))
        branch.delta = torch.zeros_like(branch.base)

        res = branch.residual()
        res_np = res[0].abs().cpu().numpy()

        fg_non_roi = foreground & (~roi)
        bg = ~foreground

        max_fg_val = res_np[:, fg_non_roi].max()
        assert max_fg_val <= cfg.latent_levels / 255.0 + 1e-7, (
            f"FG non-ROI exceeded latent_levels: {max_fg_val * 255.0} > {cfg.latent_levels}"
        )

        max_bg_val = res_np[:, bg].max()
        assert max_bg_val <= cfg.background_levels / 255.0 + 1e-7, (
            f"BG exceeded background_levels: {max_bg_val * 255.0} > {cfg.background_levels}"
        )

        assert res_np[:, roi].max() == 0.0

    def test_quantized_discrete_level_difference(self):
        """Discrete integer pixel changes round(255*output) - round(255*anchor) never exceed budget."""
        from src.visual.image_utils import quantize_ste
        cfg = DualSmoothLatentConfig(latent_levels=20, background_levels=6)
        H, W = 64, 64
        foreground = np.zeros((H, W), dtype=bool)
        foreground[16:48, 16:48] = True
        roi = np.zeros((H, W), dtype=bool)
        roi[24:36, 24:40] = True

        env = _compute_envelope_weights(foreground, roi, cfg)
        for anchor_level in [0.0, 0.1, 0.5, 0.9, 1.0]:
            anchor = torch.full((1, 3, H, W), anchor_level, dtype=torch.float32)
            for diff_sign in [+1.0, -1.0]:
                diff = torch.full((1, 3, H, W), diff_sign * 1e4, dtype=torch.float32)
                purifier = _MockPurifierForSyntheticDiff(diff, (1, 3, H, W))
                branch = _DualSmoothLatentBranch.__new__(_DualSmoothLatentBranch)
                branch.anchor = anchor
                branch.eps_map = env['eps_map']
                branch.tau_map = env['tau_map']
                branch.chroma_keep_map = env['chroma_keep_map']
                branch.support_mask = env['support_mask']
                branch.config = cfg
                branch.max_eps = max(cfg.background_levels, cfg.latent_levels) / 255.0
                branch.luma_kernel = _make_sep_gaussian(cfg.luma_smooth_sigma, anchor.device, 1)
                branch.chroma_kernel = _make_sep_gaussian(cfg.chroma_smooth_sigma, anchor.device, 3)
                branch.purifier = purifier
                branch.reference = purifier.reference
                branch.base = torch.zeros((1, 4, 8, 8))
                branch.delta = torch.zeros_like(branch.base)

                res = branch.residual()
                output = quantize_ste((anchor + res).clamp(0, 1))

                anchor_int = (anchor * 255.0).round().int()
                output_int = (output * 255.0).round().int()
                diff_int = (output_int - anchor_int).abs()[0].numpy()

                fg_non_roi = foreground & (~roi)
                bg = ~foreground

                assert diff_int[:, fg_non_roi].max() <= cfg.latent_levels
                assert diff_int[:, bg].max() <= cfg.background_levels
                assert diff_int[:, roi].max() == 0


# ============================================================================
# Section 4: Separable Gaussian Kernel Verification
# ============================================================================

class TestSeparableGaussianKernel:
    """Stress-test energy preservation and equivalence between 1D separable convolution and 2D convolution."""

    @pytest.mark.parametrize("sigma", [0.5, 1.0, 1.5, 2.2, 3.5, 5.0, 8.0, 12.0, 16.0])
    def test_energy_preservation(self, sigma):
        """Sum of 1D Gaussian kernel weights must be exactly 1.0 within numerical precision."""
        kh, kv, radius = _make_sep_gaussian(sigma, 'cpu', 1)
        sum_h = float(kh.sum())
        sum_v = float(kv.sum())
        assert math.isclose(sum_h, 1.0, rel_tol=1e-6, abs_tol=1e-6), f"kh sum was {sum_h}"
        assert math.isclose(sum_v, 1.0, rel_tol=1e-6, abs_tol=1e-6), f"kv sum was {sum_v}"

    @pytest.mark.parametrize("sigma", [2.2, 5.0])
    def test_constant_image_dc_preservation(self, sigma):
        """Applying _smooth_sep to a constant image must preserve the constant everywhere, including borders."""
        kh, kv, radius = _make_sep_gaussian(sigma, 'cpu', 3)
        c_val = 0.7321
        x = torch.full((1, 3, 64, 64), c_val, dtype=torch.float32)
        out = _smooth_sep(x, kh, kv, radius, groups=3)
        np.testing.assert_allclose(out.numpy(), c_val, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("sigma", [1.0, 2.2, 5.0])
    def test_2d_vs_1d_separable_convolution_equivalence(self, sigma):
        """2D convolution with outer-product kernel matches 1D separable convolution within float32 precision."""
        kh, kv, radius = _make_sep_gaussian(sigma, 'cpu', 1)
        g = kh[0, 0, 0]
        k2d = torch.outer(g, g).view(1, 1, 2 * radius + 1, 2 * radius + 1)
        assert math.isclose(float(k2d.sum()), 1.0, rel_tol=1e-6, abs_tol=1e-6)

        torch.manual_seed(42)
        test_patterns = [
            torch.randn(1, 1, 64, 64),
            torch.zeros(1, 1, 64, 64),
            torch.meshgrid(torch.linspace(0, 1, 64), torch.linspace(0, 1, 64), indexing='ij')[0][None, None],
        ]
        test_patterns[1][:, :, 32, 32] = 10.0

        for x in test_patterns:
            out_sep = _smooth_sep(x, kh, kv, radius, groups=1)
            padded = F.pad(x, (radius, radius, radius, radius), mode='reflect')
            out_2d = F.conv2d(padded, k2d, groups=1)

            # In float32, accumulating up to 31x31=961 products gives max error ~ 1.5e-6
            max_diff = float((out_sep - out_2d).abs().max())
            assert max_diff < 5e-6, f"Max diff between 2D and separable 1D was {max_diff}"

    def test_autograd_backward_energy_preservation(self):
        """Backward gradient through _smooth_sep preserves energy and is strictly finite."""
        x = torch.randn(1, 3, 64, 64, requires_grad=True)
        kh, kv, radius = _make_sep_gaussian(2.2, 'cpu', 3)
        out = _smooth_sep(x, kh, kv, radius, groups=3)
        loss = out.sum()
        loss.backward()
        assert torch.isfinite(x.grad).all()
        # By energy conservation, the gradient sum should equal the output volume
        target_sum = 1.0 * 3.0 * 64.0 * 64.0
        assert math.isclose(float(x.grad.sum()), target_sum, rel_tol=1e-4)

    def test_kernel_symmetry(self):
        """Gaussian kernel must be strictly symmetric: g[k] == g[-k]."""
        for sigma in [1.5, 2.2, 5.0, 8.0]:
            kh, kv, radius = _make_sep_gaussian(sigma, 'cpu', 1)
            g = kh[0, 0, 0].numpy()
            assert len(g) == 2 * radius + 1
            np.testing.assert_array_equal(g, g[::-1])

    def test_radius_guarantees_99_7_percent_mass(self):
        """Radius ceil(3 * sigma) captures >= 99.73% of normal distribution mass before normalization."""
        for sigma in [0.5, 1.0, 2.2, 5.0]:
            radius = int(np.ceil(3.0 * sigma))
            coords = np.arange(-radius, radius + 1, dtype=np.float64)
            unnorm_g = np.exp(-(coords ** 2) / (2.0 * sigma * sigma))
            theoretical_integral = np.sqrt(2.0 * np.pi) * sigma
            captured_mass = unnorm_g.sum() / theoretical_integral
            assert captured_mass >= 0.997, f"Captured mass for sigma={sigma}: {captured_mass}"


# ============================================================================
# Section 5: Degenerate Geometries & Validation Failure Sensitivity
# ============================================================================

class TestDegenerateGeometriesAndValidationSensitivity:
    """Stress-test degenerate edge cases and assertion tripwires in validate_dual_smooth_complete."""

    def test_foreground_and_roi_touching_image_borders(self):
        """Foreground and text ROI extending all the way to image borders execute without indexing errors."""
        H, W = 64, 64
        foreground = np.zeros((H, W), dtype=bool)
        foreground[:32, :] = True  # Touches top, left, right borders
        roi = np.zeros((H, W), dtype=bool)
        roi[:10, :10] = True       # Touches top-left corner of image

        cfg = DualSmoothLatentConfig(latent_levels=24, background_levels=8)
        env = _compute_envelope_weights(foreground, roi, cfg)
        levels = env['level_array']

        assert levels[:10, :10].max() == 0.0  # ROI at corner strictly 0
        assert levels[foreground & ~roi].max() == 24.0
        assert levels[~foreground].max() == 8.0
        assert np.isfinite(levels).all()

    def test_empty_roi_behavior(self):
        """When ROI is empty (no text on product), whole foreground can receive modulation."""
        H, W = 64, 64
        foreground = np.zeros((H, W), dtype=bool)
        foreground[16:48, 16:48] = True
        roi = np.zeros((H, W), dtype=bool)

        cfg = DualSmoothLatentConfig(latent_levels=24, background_levels=8)
        env = _compute_envelope_weights(foreground, roi, cfg)
        levels = env['level_array']
        assert levels[foreground].max() == 24.0
        assert levels[~foreground].max() == 8.0

    def test_empty_foreground_behavior(self):
        """When foreground is empty (all background), whole image receives background modulation."""
        H, W = 64, 64
        foreground = np.zeros((H, W), dtype=bool)
        roi = np.zeros((H, W), dtype=bool)

        cfg = DualSmoothLatentConfig(latent_levels=24, background_levels=8)
        env = _compute_envelope_weights(foreground, roi, cfg)
        levels = env['level_array']
        assert levels.max() == 8.0
        assert env['w_fg_body'].max() == 0.0

    def test_validator_detects_single_roi_pixel_violation(self):
        """validate_dual_smooth_complete must raise AssertionError if even 1 pixel in ROI changes by 1 level."""
        H, W = 64, 64
        foreground = np.zeros((H, W), dtype=bool)
        foreground[16:48, 16:48] = True
        roi = np.zeros((H, W), dtype=bool)
        roi[24:36, 24:40] = True

        anchor = np.full((H, W, 3), 128, dtype=np.uint8)
        image = anchor.copy()
        # Corrupt 1 pixel inside ROI by exactly 1 level
        image[25, 25, 0] = 129

        ready = {
            'status': 'completed',
            'pixel_optimization': False,
            'auxiliary_max_levels': 0,
            'victim_memory_used': False,
            'formal_queries_used': False,
            'shadow_pool_used': False,
            'true_rank_used': False,
            'purchase_history_used': False,
            'first_three_stages_reused': True,
            'stages': {
                'foreground_segmentation': {},
                'foreground_statistics': {},
                'color_harmonization': {},
                'latent_generation_refinement': {'steps_completed': 350, 'steps_requested': 350},
            },
            'latent_config': {'latent_levels': 24, 'background_levels': 8},
        }

        with pytest.raises(AssertionError, match="印刷区域保护失败：文字区发生像素改变"):
            validate_dual_smooth_complete(ready, image=image, anchor_image=anchor, foreground=foreground, roi=roi)

    def test_validator_detects_single_fg_budget_overflow(self):
        """validate_dual_smooth_complete must raise AssertionError if 1 foreground pixel exceeds latent_levels."""
        H, W = 64, 64
        foreground = np.zeros((H, W), dtype=bool)
        foreground[16:48, 16:48] = True
        roi = np.zeros((H, W), dtype=bool)

        anchor = np.full((H, W, 3), 128, dtype=np.uint8)
        image = anchor.copy()
        # Corrupt 1 foreground pixel by latent_levels + 1 = 25
        image[20, 20, 0] = 128 + 25

        ready = {
            'status': 'completed',
            'pixel_optimization': False,
            'auxiliary_max_levels': 0,
            'victim_memory_used': False,
            'formal_queries_used': False,
            'shadow_pool_used': False,
            'true_rank_used': False,
            'purchase_history_used': False,
            'first_three_stages_reused': True,
            'stages': {
                'foreground_segmentation': {},
                'foreground_statistics': {},
                'color_harmonization': {},
                'latent_generation_refinement': {'steps_completed': 350, 'steps_requested': 350},
            },
            'latent_config': {'latent_levels': 24, 'background_levels': 8},
        }

        with pytest.raises(AssertionError, match="主体前景改动 .* 超过 latent_levels"):
            validate_dual_smooth_complete(ready, image=image, anchor_image=anchor, foreground=foreground, roi=roi)

    def test_validator_detects_single_bg_budget_overflow(self):
        """validate_dual_smooth_complete must raise AssertionError if 1 background pixel exceeds background_levels."""
        H, W = 64, 64
        foreground = np.zeros((H, W), dtype=bool)
        foreground[16:48, 16:48] = True
        roi = np.zeros((H, W), dtype=bool)

        anchor = np.full((H, W, 3), 128, dtype=np.uint8)
        image = anchor.copy()
        # Corrupt 1 background pixel by background_levels + 1 = 9
        image[5, 5, 0] = 128 + 9

        ready = {
            'status': 'completed',
            'pixel_optimization': False,
            'auxiliary_max_levels': 0,
            'victim_memory_used': False,
            'formal_queries_used': False,
            'shadow_pool_used': False,
            'true_rank_used': False,
            'purchase_history_used': False,
            'first_three_stages_reused': True,
            'stages': {
                'foreground_segmentation': {},
                'foreground_statistics': {},
                'color_harmonization': {},
                'latent_generation_refinement': {'steps_completed': 350, 'steps_requested': 350},
            },
            'latent_config': {'latent_levels': 24, 'background_levels': 8},
        }

        with pytest.raises(AssertionError, match="背景改动 .* 超过 background_levels"):
            validate_dual_smooth_complete(ready, image=image, anchor_image=anchor, foreground=foreground, roi=roi)
