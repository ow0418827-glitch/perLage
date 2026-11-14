"""仅供洗发水冻结检索复测的本地原生巨型编码器，不改变共享加载器。"""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace


MODEL_NAME = "ViT-g-14"
PRETRAINED = "laion2b_s12b_b42k"
REPOSITORY = "laion/CLIP-ViT-g-14-laion2B-s12B-b42K"


def _sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _revision(path):
    parts = Path(path).parts
    try:
        return parts[parts.index("snapshots") + 1]
    except (ValueError, IndexError):
        return None


class _NativeG14:
    """提供现有完整分块适配器所需的基础编码接口。"""

    def __init__(self, config_path, device):
        import open_clip
        from open_clip.pretrained import get_pretrained_cfg
        from safetensors.torch import load_file
        import torch

        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        path = Path(config_path).resolve()
        settings = json.loads(path.read_text(encoding="utf-8-sig"))
        if settings.get("setting") != "held_out_public_encoder":
            raise ValueError("本轮巨型检索器必须明确登记为独立留出评估编码器")
        spec = settings["surrogate"]
        identity = (spec.get("name"), spec.get("pretrained"), spec.get("hf_repo"))
        if identity != (MODEL_NAME, PRETRAINED, REPOSITORY):
            raise ValueError("本轮只允许冻结的巨型模型及预训练身份")
        weight = Path(spec["weight_path"])
        if not weight.is_absolute():
            weight = path.parent / weight
        weight = weight.resolve()
        if not weight.is_file() or weight.suffix != ".safetensors":
            raise FileNotFoundError("须显式提供本地安全张量权重，不下载或回退")
        expected = settings.get("expected_weight_sha256")
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError("冻结编码器配置缺少完整权重摘要")
        weight_sha = _sha256(weight)
        if weight_sha != expected:
            raise ValueError("实际巨型编码器权重摘要与本轮冻结配置不同")

        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("本轮须使用显卡，不允许静默切换处理器")
        if self.device.index is not None and self.device.index >= torch.cuda.device_count():
            raise RuntimeError("本轮显卡设备编号不存在")

        architecture = copy.deepcopy(open_clip.get_model_config(MODEL_NAME))
        registered = copy.deepcopy(get_pretrained_cfg(MODEL_NAME, PRETRAINED))
        if not architecture or not registered:
            raise ValueError("安装的编码库缺少所需模型结构或预训练注册信息")
        if architecture.get("quick_gelu", False) or registered.get("quick_gelu", False):
            raise ValueError("冻结巨型模型的本地原生激活注册与预期不符")
        mean = tuple(float(v) for v in registered["mean"])
        std = tuple(float(v) for v in registered["std"])
        if len(mean) != 3 or len(std) != 3 or not all(math.isfinite(v) for v in (*mean, *std)) or min(std) <= 0:
            raise ValueError("预训练图像归一化参数无效")

        # 无预训练下载入口，严格使用该架构原生普通激活；不调用项目中
        # 对所有模型强制快速激活的共享构造器。
        model = open_clip.create_model(MODEL_NAME, pretrained=None, force_quick_gelu=False)
        state = load_file(str(weight), device="cpu")
        if not isinstance(state, dict) or not state or not all(isinstance(k, str) and torch.is_tensor(v) for k, v in state.items()):
            raise ValueError("本地权重不是有效的非空模型参数字典")
        state = {key.removeprefix("module."): value for key, value in state.items()}
        model.load_state_dict(state, strict=True)
        del state
        actual = {name: sum(type(module).__name__ == name for module in model.modules())
                  for name in ("QuickGELU", "GELU")}
        if actual != {"QuickGELU": 0, "GELU": 64}:
            raise RuntimeError("巨型模型实际激活模块不符合四十层视觉加二十四层文字的原生结构")
        model.to(device=self.device, dtype=torch.float32).eval().requires_grad_(False)
        self.model = model
        self.tokenizer = open_clip.get_tokenizer(MODEL_NAME)
        self.context_length = int(getattr(self.tokenizer, "context_length", 77))
        if self.context_length != 77 or not callable(getattr(self.tokenizer, "encode", None)):
            raise ValueError("巨型编码器分词器或文字上下文不符合预期")
        self.surrogate = SimpleNamespace(model=self.model, tokenizer=self.tokenizer)
        size = self.model.visual.image_size
        self.image_size = int(size if isinstance(size, int) else size[0])
        if self.image_size != 224 or not hasattr(self.model.visual, "forward_intermediates"):
            raise ValueError("巨型模型图像尺寸或全局及局部特征接口不符合预期")
        self.mean = torch.tensor(mean, device=self.device, dtype=torch.float32)[None, :, None, None]
        self.std = torch.tensor(std, device=self.device, dtype=torch.float32)[None, :, None, None]

        package = Path(open_clip.__file__).resolve().parent
        source_paths = {
            "native_model_config": package / "model_configs" / (MODEL_NAME + ".json"),
            "pretrained_registration": package / "pretrained.py",
            "model_factory": package / "factory.py",
            "evaluation_wrapper": Path(__file__).resolve(),
            "chunked_encoder": Path(__file__).resolve().parents[1] / "src/visual/encoders/joint.py",
        }
        source_hashes = {name: dict(path=str(source), sha256=_sha256(source))
                         for name, source in source_paths.items()}
        loading = dict(architecture=architecture, quick_gelu=False, activation="GELU",
                       mean=list(mean), std=list(std), metadata_source="native_registered_pretrained_tag",
                       registered_preprocess={key: registered[key] for key in ("interpolation", "resize_mode") if key in registered},
                       strict_state_dict=True, actual_activation_modules=actual,
                       loader_version="shampoo_native_g14_v1", source_hashes=source_hashes)
        surrogate = dict(role="held_out_evaluation_encoder", name=MODEL_NAME, pretrained=PRETRAINED,
                         hf_repo=REPOSITORY, weight_path=str(weight), weight_sha256=weight_sha,
                         weight_revision=_revision(weight), device=str(self.device), precision="fp32",
                         image_size=self.image_size, preprocess=dict(resize="bicubic_direct_square",
                             align_corners=False, antialias=True, mean=list(mean), std=list(std)),
                         open_clip_torch_version=importlib.metadata.version("open_clip_torch"),
                         requires_true_local_tokens=True, loading=loading)
        surrogate["cache_identity"] = hashlib.sha256(json.dumps(surrogate, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        self.snapshot = dict(provider="clip", loader="native_local_registered_g14", surrogate_number=None,
                             surrogate=surrogate, fusion="normalized_image_text_sum",
                             text_policy="reject_over_context_without_chunked_adapter",
                             encoder_config_sha256=_sha256(path),
                             evaluation_protocol=dict(setting="held_out_public_encoder",
                                 matches_training_surrogate=False, matching_training_surrogate_indices=[],
                                 black_box_transfer_claimed=True,
                                 note="Retrieval-only encoder substitution; no shopping transfer claim."))

    def encode(self, path):
        import numpy as np
        from PIL import Image
        import torch
        import torch.nn.functional as functional

        with Image.open(path) as image:
            array = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device=self.device, dtype=torch.float32) / 255
        with torch.no_grad():
            if tensor.shape[-2:] != (self.image_size, self.image_size):
                tensor = functional.interpolate(tensor, size=(self.image_size, self.image_size),
                    mode="bicubic", align_corners=False, antialias=True)
            output = self.model.visual.forward_intermediates((tensor - self.mean) / self.std,
                indices=[-1], output_fmt="NLC", output_extra_tokens=True)
            if not isinstance(output, dict) or "image_features" not in output or "image_intermediates" not in output:
                raise RuntimeError("巨型模型没有返回真实全局及局部特征")
            patches = output["image_intermediates"][-1]
            if patches.ndim != 3 or patches.shape[1] <= 1:
                raise RuntimeError("巨型模型局部特征接口没有返回真实图像块")
            value = functional.normalize(output["image_features"].float(), dim=-1)
            if value.ndim != 2 or value.shape[0] != 1 or not torch.isfinite(value).all():
                raise RuntimeError("巨型模型返回无效全局图像向量")
            return value[0].cpu().numpy()


def make_encoder(config_path, device="cuda"):
    """返回现有完整分块适配器；只在实际调用时构造并载入模型。"""
    from src.visual.encoders.joint import ChunkedClipEncoder
    return ChunkedClipEncoder(_NativeG14(config_path, device))
