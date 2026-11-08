"""Empirical stress test suite for Milestone M1: QuickGELU enforcement & baseline preservation.
Executed by Challenger 1 (challenger_m1_1).
"""
import hashlib
import io
import math
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import open_clip
import torch
import torch.nn as nn

from src.paths import PROJECT
from src.visual.encoders.backbone import (
    BackboneConfig,
    CLIP_MEAN,
    CLIP_STD,
    ImageFeatures,
    OpenCLIPSurrogate,
    load_local_clip_model,
    pretrained_settings,
    resolve_hf_cached_weight,
)


class TestPretrainedSettingsStress(unittest.TestCase):
    def test_invalid_model_architecture(self):
        with self.assertRaises(ValueError) as ctx:
            pretrained_settings("NonExistentModel_XYZ123", "openai")
        self.assertIn("未知视觉编码器结构", str(ctx.exception))

    def test_valid_model_forces_quickgelu_settings(self):
        for model_name, pretrained in [
            ("ViT-B-32", "laion2b_s34b_b79k"),
            ("ViT-B-32", "openai"),
            ("ViT-g-14", "laion2b_s12b_b42k"),
        ]:
            settings = pretrained_settings(model_name, pretrained)
            self.assertTrue(settings["quick_gelu"], f"{model_name} quick_gelu must be True")
            self.assertEqual(settings["activation"], "QuickGELU", f"{model_name} activation must be 'QuickGELU'")
            self.assertEqual(len(settings["mean"]), 3)
            self.assertEqual(len(settings["std"]), 3)

    def test_unregistered_pretrained_tag_fallbacks_to_quickgelu(self):
        settings = pretrained_settings("ViT-B-32", "unregistered_custom_tag_999")
        self.assertTrue(settings["quick_gelu"])
        self.assertEqual(settings["activation"], "QuickGELU")
        self.assertEqual(settings["mean"], CLIP_MEAN)
        self.assertEqual(settings["std"], CLIP_STD)
        self.assertEqual(settings["metadata_source"], "explicit_configuration")


class TestGELUEnforcementAndFailFast(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w_b32 = resolve_hf_cached_weight("laion/CLIP-ViT-B-32-laion2B-s34B-b79K")
        assert cls.w_b32 and cls.w_b32.is_file(), f"Missing ViT-B-32 weight at {cls.w_b32}"

    def test_missing_weight_file_raises_filenotfound(self):
        with self.assertRaises(FileNotFoundError):
            load_local_clip_model("ViT-B-32", "laion2b_s34b_b79k", "non_existent_weight_path_12345.bin")

    def test_invalid_weight_format_raises_valueerror(self):
        bad_weight_path = PROJECT / "output" / "tmp_bad_weight.pt"
        bad_weight_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"unexpected_string": 12345}, bad_weight_path)
        try:
            with self.assertRaises(ValueError):
                load_local_clip_model("ViT-B-32", "laion2b_s34b_b79k", bad_weight_path)
        finally:
            if bad_weight_path.exists():
                bad_weight_path.unlink()

    def test_standard_gelu_instantiation_raises_runtimeerror(self):
        """Mock open_clip.create_model to create standard GELU (force_quick_gelu=False).
        load_local_clip_model MUST detect GELU > 0 and raise RuntimeError.
        """
        original_create_model = open_clip.create_model

        def mock_create_model(name, pretrained=None, force_quick_gelu=True, **kwargs):
            # Adversarially override force_quick_gelu to False to generate standard GELU
            return original_create_model(name, pretrained=pretrained, force_quick_gelu=False, **kwargs)

        with patch("open_clip.create_model", side_effect=mock_create_model):
            with self.assertRaises(RuntimeError) as ctx:
                load_local_clip_model("ViT-B-32", "laion2b_s34b_b79k", self.w_b32)
            self.assertIn("未收敛至纯 QuickGELU 激活函数", str(ctx.exception))

    def test_tampered_gelu_module_injection_caught(self):
        """Simulate tampering: replace a QuickGELU module with standard nn.GELU.
        The audit check actual['QuickGELU'] > 0 and actual.get('GELU', 0) == 0 MUST catch it.
        """
        original_create_model = open_clip.create_model

        def mock_create_model_with_single_gelu(*args, **kwargs):
            model = original_create_model(*args, **kwargs)
            # Replace one QuickGELU in transformer resblocks with standard GELU
            block0 = model.visual.transformer.resblocks[0]
            block0.mlp.act = nn.GELU()
            return model

        with patch("open_clip.create_model", side_effect=mock_create_model_with_single_gelu):
            with self.assertRaises(RuntimeError) as ctx:
                load_local_clip_model("ViT-B-32", "laion2b_s34b_b79k", self.w_b32)
            self.assertIn("未收敛至纯 QuickGELU 激活函数", str(ctx.exception))

    def test_zero_activation_modules_caught(self):
        """If a model had no QuickGELU modules at all, it must also fail."""
        original_create_model = open_clip.create_model

        def mock_create_model_no_act(*args, **kwargs):
            model = original_create_model(*args, **kwargs)
            # Replace all QuickGELU with nn.Identity
            for m in model.modules():
                if type(m).__name__ == "QuickGELU":
                    m.__class__ = nn.Identity
            return model

        with patch("open_clip.create_model", side_effect=mock_create_model_no_act):
            with self.assertRaises(RuntimeError) as ctx:
                load_local_clip_model("ViT-B-32", "laion2b_s34b_b79k", self.w_b32)
            self.assertIn("未收敛至纯 QuickGELU 激活函数", str(ctx.exception))


class TestLiveForwardInferenceStress(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w_b32 = resolve_hf_cached_weight("laion/CLIP-ViT-B-32-laion2B-s34B-b79K")
        cls.w_g14 = resolve_hf_cached_weight(
            "laion/CLIP-ViT-g-14-laion2B-s12B-b42K",
            filename="open_clip_pytorch_model.safetensors",
        )
        assert cls.w_b32 and cls.w_b32.is_file()
        assert cls.w_g14 and cls.w_g14.is_file()

        cls.model_b32, cls.settings_b32 = load_local_clip_model("ViT-B-32", "laion2b_s34b_b79k", cls.w_b32)
        cls.model_g14, cls.settings_g14 = load_local_clip_model("ViT-g-14", "laion2b_s12b_b42k", cls.w_g14)

    def test_module_counts_strict(self):
        actual_b32 = {name: sum(type(m).__name__ == name for m in self.model_b32.modules()) for name in ("QuickGELU", "GELU")}
        self.assertEqual(actual_b32["QuickGELU"], 24)
        self.assertEqual(actual_b32.get("GELU", 0), 0)

        actual_g14 = {name: sum(type(m).__name__ == name for m in self.model_g14.modules()) for name in ("QuickGELU", "GELU")}
        self.assertEqual(actual_g14["QuickGELU"], 64)
        self.assertEqual(actual_g14.get("GELU", 0), 0)

    def test_vit_b32_forward_varied_batches_and_dtypes(self):
        seeds = [42, 123, 999]
        # Test CPU with batch sizes [1, 2]
        self.model_b32.to("cpu").float().eval()
        for b in [1, 2]:
            for seed in seeds:
                torch.manual_seed(seed)
                x = torch.randn(b, 3, 224, 224, device="cpu", dtype=torch.float32)
                with torch.no_grad():
                    feats = self.model_b32.encode_image(x)
                self.assertEqual(feats.shape, (b, 512))
                self.assertTrue(torch.isfinite(feats).all().item())

        # Test CUDA with batch sizes [1, 2, 7, 16] and float32/float16
        if torch.cuda.is_available():
            for dtype in [torch.float32, torch.float16]:
                self.model_b32.to("cuda")
                if dtype == torch.float16:
                    self.model_b32.half()
                else:
                    self.model_b32.float()
                self.model_b32.eval()
                for b in [1, 2, 7, 16]:
                    for seed in seeds:
                        torch.manual_seed(seed)
                        x = torch.randn(b, 3, 224, 224, device="cuda", dtype=dtype)
                        with torch.no_grad():
                            feats = self.model_b32.encode_image(x)
                        self.assertEqual(feats.shape, (b, 512))
                        self.assertTrue(torch.isfinite(feats).all().item())

    def test_vit_g14_forward_varied_batches_and_dtypes(self):
        seeds = [42, 123, 999]
        # Test CPU with batch size 1 (lightweight)
        self.model_g14.to("cpu").float().eval()
        torch.manual_seed(42)
        x = torch.randn(1, 3, 224, 224, device="cpu", dtype=torch.float32)
        with torch.no_grad():
            feats = self.model_g14.encode_image(x)
        self.assertEqual(feats.shape, (1, 1024))
        self.assertTrue(torch.isfinite(feats).all().item())

        # Test CUDA with batch sizes [1, 2, 7, 16] and float32/float16
        if torch.cuda.is_available():
            for dtype in [torch.float32, torch.float16]:
                self.model_g14.to("cuda")
                if dtype == torch.float16:
                    self.model_g14.half()
                else:
                    self.model_g14.float()
                self.model_g14.eval()
                for b in [1, 2, 7, 16]:
                    for seed in seeds:
                        torch.manual_seed(seed)
                        x = torch.randn(b, 3, 224, 224, device="cuda", dtype=dtype)
                        with torch.no_grad():
                            feats = self.model_g14.encode_image(x)
                        self.assertEqual(feats.shape, (b, 1024))
                        self.assertTrue(torch.isfinite(feats).all().item())

    def test_surrogate_wrapper_features_and_patch_tokens(self):
        cfg = BackboneConfig(
            name="ViT-B-32",
            pretrained="laion2b_s34b_b79k",
            hf_repo="laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        surrogate = OpenCLIPSurrogate(cfg, device=device, allow_device_fallback=True)

        for b in [1, 3]:
            imgs = torch.randn(b, 3, 224, 224)
            features = surrogate.encode_image(imgs)
            self.assertIsInstance(features, ImageFeatures)
            self.assertEqual(features.global_embedding.shape, (b, 512))
            self.assertEqual(features.patch_tokens.shape, (b, 49, 768))
            # Test global embedding normalization
            norms = torch.norm(features.global_embedding, dim=-1)
            self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-4))

    def test_surrogate_text_encoding_and_context_overflow(self):
        cfg = BackboneConfig(
            name="ViT-B-32",
            pretrained="laion2b_s34b_b79k",
            hf_repo="laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        surrogate = OpenCLIPSurrogate(cfg, device=device, allow_device_fallback=True)

        # Normal text
        text_emb = surrogate.encode_text(["a photo of shampoo", "a chair"])
        self.assertEqual(text_emb.shape, (2, 512))

        # Context length overflow check (context length is 77 tokens)
        long_text = "word " * 100
        with self.assertRaises(ValueError) as ctx:
            surrogate.encode_text(long_text)
        self.assertIn("文字目标超过模型容量", str(ctx.exception))


class TestHistoricalBaselineIntegrity(unittest.TestCase):
    def test_baseline_images_byte_match(self):
        lage_base = Path("d:/Material/Literature Reading/code/lage")
        pairs = [
            (
                PROJECT / "output/shampoo_ablation_20261107/images/lucid_original_16.png",
                lage_base / "src/output/lage_runs/lucid_shampoo_full_retest_20261027/groups/07_shampoo/attack/adversarial.png",
                "8f26a32627fcc6dee11ae18f4a67e7c320e3756a71ae2a57d7dfde96bd451852",
                "Historical lucid 16-step from lage",
            ),
            (
                PROJECT / "output/shampoo_ablation_20261107/images/perlage_original_no_shadow.png",
                lage_base / "src/output/lage_runs/catalog_shampoo_no_shadow_20261029/generated/07_shampoo/adversarial.png",
                "69f7f19d6bef591dbe933b78f3945dd88f3171ee306dd7b0c32e905fc5fc6e80",
                "Historical perLage no-shadow from lage",
            ),
            (
                PROJECT / "output/shampoo_ablation_20261107/images/perlage_legacy_reference.png",
                lage_base / "src/output/lage_runs/catalog_shampoo_20261028/generated/07_shampoo/adversarial.png",
                "42c312c4a2216a759dcf9acef8d0e4936fc318d20bd2b87384173b66bd241529",
                "Historical perLage legacy reference from lage",
            ),
            (
                PROJECT / "output/shampoo_ablation_20261107/images/perlage_color_revision_reference.png",
                PROJECT / "output/color_full_pipeline_20261106/runs/dominant/07_shampoo/generation/adversarial.png",
                "8b25b25f3d0dbf3f4e0a6733183857f165939cb00e38db14092a1dda3c2b76bb",
                "Historical color revision reference from perLage",
            ),
        ]

        for local_p, ref_p, expected_sha, desc in pairs:
            self.assertTrue(local_p.is_file(), f"Local image missing: {local_p}")
            self.assertTrue(ref_p.is_file(), f"Reference image missing: {ref_p}")
            local_bytes = local_p.read_bytes()
            ref_bytes = ref_p.read_bytes()
            self.assertEqual(local_bytes, ref_bytes, f"Byte mismatch for {desc} ({local_p.name})")
            actual_sha = hashlib.sha256(local_bytes).hexdigest()
            self.assertEqual(actual_sha, expected_sha, f"SHA256 mismatch for {desc} ({local_p.name})")


if __name__ == "__main__":
    unittest.main(verbosity=2)
