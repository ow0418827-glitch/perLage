from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import json
import math
import os
from src.paths import ProjectPath as Path
from typing import Any, Iterable, Mapping

import torch
import torch.nn.functional as F

@dataclass(frozen=True)
class BackboneConfig:
    name: str
    pretrained: str
    hf_repo: str | None = None
    weight_path: str | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "BackboneConfig":
        return cls(
            name=str(raw["name"]),
            pretrained=str(raw.get("pretrained", "openai")),
            hf_repo=str(raw["hf_repo"]) if raw.get("hf_repo") else None,
            weight_path=str(raw["weight_path"]) if raw.get("weight_path") else None,
        )
from src.visual.io import sha256_file, stable_key


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclass(frozen=True)
class ImageFeatures:
    global_embedding: torch.Tensor
    patch_tokens: torch.Tensor


def resolve_device(requested: str, *, allow_fallback: bool = False) -> torch.device:
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        if allow_fallback:
            return torch.device("cpu")
        raise RuntimeError("配置要求使用图形处理器，但当前张量框架未检测到可用设备。")
    if device.type == "cuda" and device.index is not None:
        if device.index >= torch.cuda.device_count():
            if allow_fallback:
                return torch.device("cpu")
            raise RuntimeError(f"配置要求的图形处理器序号不存在：{device.index}")
    return device


def _huggingface_hub_roots(cache_dir: str | Path | None = None) -> list[Path]:
    roots: list[Path] = []
    if cache_dir:
        candidate = Path(cache_dir)
        roots.append(candidate if candidate.name == "hub" else candidate / "hub")
    for variable in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if os.environ.get(variable):
            roots.append(Path(os.environ[variable]))
    if os.environ.get("HF_HOME"):
        roots.append(Path(os.environ["HF_HOME"]) / "hub")
    if os.environ.get("TRANSFORMERS_CACHE"):
        roots.append(Path(os.environ["TRANSFORMERS_CACHE"]))
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root.resolve()) if root.exists() else str(root)
        if key not in seen:
            unique.append(root)
            seen.add(key)
    return unique


def resolve_hf_cached_weight(
    hf_repo: str | None,
    filename: str = "open_clip_model.safetensors",
    cache_dir: str | Path | None = None,
) -> Path | None:
    if not hf_repo:
        return None
    repository_name = f"models--{hf_repo.replace('/', '--')}"
    matches: list[Path] = []
    for hub_root in _huggingface_hub_roots(cache_dir):
        repository = hub_root / repository_name
        if not repository.exists():
            continue
        main_ref = repository / "refs" / "main"
        if main_ref.is_file():
            revision = main_ref.read_text(encoding="utf-8").strip()
            candidate = repository / "snapshots" / revision / filename
            if candidate.is_file():
                return candidate
        snapshots = repository / "snapshots"
        if snapshots.exists():
            matches.extend(path for path in snapshots.glob(f"*/{filename}") if path.is_file())
    return sorted(matches, key=str)[-1] if matches else None


def resolve_hf_snapshot(hf_repo: str, cache_dir: str | Path | None = None) -> Path | None:
    repository_name = f"models--{hf_repo.replace('/', '--')}"
    candidates: list[Path] = []
    for hub_root in _huggingface_hub_roots(cache_dir):
        repository = hub_root / repository_name
        main_ref = repository / "refs" / "main"
        if main_ref.is_file():
            revision = main_ref.read_text(encoding="utf-8").strip()
            snapshot = repository / "snapshots" / revision
            if snapshot.is_dir():
                return snapshot.resolve()
        snapshots = repository / "snapshots"
        if snapshots.is_dir():
            candidates.extend(path.resolve() for path in snapshots.iterdir() if path.is_dir())
    return sorted(candidates, key=str)[-1] if candidates else None


def _infer_image_size(model: torch.nn.Module) -> int:
    image_size = getattr(getattr(model, "visual", None), "image_size", 224)
    if isinstance(image_size, Iterable) and not isinstance(image_size, (str, bytes)):
        return int(tuple(image_size)[0])
    return int(image_size)


def _revision_from_weight_path(path: Path | None) -> str | None:
    if path is None:
        return None
    parts = path.parts
    try:
        index = parts.index("snapshots")
    except ValueError:
        return None
    return parts[index + 1] if index + 1 < len(parts) else None

def _is_safetensors_file(path: str | Path | None) -> bool:
    if not path:
        return False
    path_obj = Path(path)
    if path_obj.name.endswith(".safetensors") or str(path).endswith(".safetensors"):
        return True
    try:
        if path_obj.is_file():
            with open(path_obj, "rb") as f:
                header = f.read(9)
            if len(header) >= 9:
                import struct
                header_size = struct.unpack("<Q", header[:8])[0]
                if 0 < header_size < 100 * 1024 * 1024 and header[8:9] == b"{":
                    return True
    except Exception:
        pass
    return False


def pretrained_settings(model_name, pretrained):
    """本地权重文件也须继承其声明的预训练结构，不能只继承模型名。"""
    import open_clip
    from open_clip.pretrained import get_pretrained_cfg
    architecture = open_clip.get_model_config(model_name)
    if architecture is None:
        raise ValueError('未知视觉编码器结构：' + model_name)
    registered = get_pretrained_cfg(model_name, pretrained)
    mean = tuple(float(v) for v in (registered.get('mean', CLIP_MEAN) if registered else CLIP_MEAN))
    std = tuple(float(v) for v in (registered.get('std', CLIP_STD) if registered else CLIP_STD))
    if len(mean) != 3 or len(std) != 3 or not all(math.isfinite(v) for v in (*mean, *std)) or min(std) <= 0:
        raise ValueError('预训练图像归一化参数无效')
    return dict(architecture=architecture, quick_gelu=True,
                activation='QuickGELU', mean=mean, std=std,
                metadata_source='registered_pretrained_tag' if registered else 'explicit_configuration',
                registered_preprocess={k: registered[k] for k in ('interpolation', 'resize_mode') if registered and k in registered})


def load_local_clip_model(model_name, pretrained, weight_path):
    """只创建结构并严格载入本地参数；拒绝缺失参数、额外参数和不安全回退。"""
    import open_clip
    settings = pretrained_settings(model_name, pretrained)
    path = Path(weight_path)
    if not path.is_file():
        raise FileNotFoundError('本地视觉编码器权重不存在：' + str(path))
    if _is_safetensors_file(path):
        from safetensors.torch import load_file
        state = load_file(str(path), device='cpu')
    else:
        state = torch.load(path, map_location='cpu', weights_only=True)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    if not isinstance(state, dict) or not state or not all(isinstance(k, str) and torch.is_tensor(v) for k,v in state.items()):
        raise ValueError('视觉编码器权重必须是非空参数字典')
    state = {k.removeprefix('module.'): v for k,v in state.items()}
    model = open_clip.create_model(model_name, pretrained=None,
                                  force_quick_gelu=True)
    model.load_state_dict(state, strict=True)
    actual = {name: sum(type(m).__name__ == name for m in model.modules())
              for name in ('QuickGELU', 'GELU')}
    if not (actual['QuickGELU'] > 0 and actual.get('GELU', 0) == 0):
        raise RuntimeError('视觉编码器未收敛至纯 QuickGELU 激活函数')
    settings.update(strict_state_dict=True, actual_activation_modules=actual,
                    loader_version=2, loader_sha256=sha256_file(__file__))
    return model, settings


class OpenCLIPSurrogate:
    """参考项目代理模型的独立兼容包装，并显式拒绝局部特征静默回退。"""

    def __init__(
        self,
        config: BackboneConfig,
        *,
        device: str = "cuda",
        precision: str = "fp32",
        allow_downloads: bool = False,
        allow_device_fallback: bool = False,
        require_true_local_tokens: bool = True,
        hash_weight: bool = True,
    ) -> None:
        try:
            import open_clip
        except ImportError as exc:
            raise RuntimeError(
                f"无法导入 open_clip_torch 或其运行依赖：{exc}"
            ) from exc
        if precision != "fp32":
            raise ValueError("首轮攻击代理只允许 fp32 精度。")
        self.config = config
        self.device = resolve_device(device, allow_fallback=allow_device_fallback)
        self.precision = precision
        self.require_true_local_tokens = bool(require_true_local_tokens)

        weight_path: Path | None = None
        if config.weight_path:
            weight_path = Path(config.weight_path)
            if not weight_path.is_file():
                raise FileNotFoundError(f"显式指定的代理权重不存在：{weight_path}")
        else:
            weight_path = resolve_hf_cached_weight(config.hf_repo)
        if weight_path is None:
            if not allow_downloads:
                raise FileNotFoundError(
                    f"没有找到 {config.name}/{config.pretrained} 的本地冻结权重；"
                    "当前配置禁止自动下载。请先显式准备权重。")
            from open_clip.pretrained import download_pretrained, get_pretrained_cfg
            registered = get_pretrained_cfg(config.name, config.pretrained)
            if not registered:
                raise ValueError('无法下载未知预训练标签')
            weight_path = Path(download_pretrained(registered))
        self.model, self.loading = load_local_clip_model(
            config.name, config.pretrained, weight_path)
        self.model.eval().to(self.device, dtype=torch.float32)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.tokenizer = open_clip.get_tokenizer(config.name)
        self.image_size = _infer_image_size(self.model)
        self.mean = torch.tensor(
            self.loading['mean'], device=self.device, dtype=torch.float32
        ).view(1, 3, 1, 1)
        self.std = torch.tensor(
            self.loading['std'], device=self.device, dtype=torch.float32
        ).view(1, 3, 1, 1)
        self.weight_path = weight_path
        self.weight_sha256 = sha256_file(weight_path) if weight_path and hash_weight else None
        self.weight_revision = _revision_from_weight_path(weight_path)
        self._open_clip_version = importlib.metadata.version("open-clip-torch")
        self._cache_identity = stable_key(
            (
                config.name,
                config.pretrained,
                config.hf_repo or "",
                str(weight_path or "downloaded-by-open-clip"),
                self.weight_sha256 or "hash-not-recorded",
                self.weight_revision or "revision-unavailable",
                precision,
                str(self.image_size),
                json.dumps(self.loading, sort_keys=True),
                self._open_clip_version,
            )
        )

    @property
    def cache_identity(self) -> str:
        return self._cache_identity

    def snapshot(self) -> dict[str, Any]:
        return {
            "role": "attack_surrogate",
            "name": self.config.name,
            "pretrained": self.config.pretrained,
            "hf_repo": self.config.hf_repo,
            "weight_path": str(self.weight_path) if self.weight_path else None,
            "weight_sha256": self.weight_sha256,
            "weight_revision": self.weight_revision,
            "device": str(self.device),
            "precision": self.precision,
            "image_size": self.image_size,
            "preprocess": {
                "resize": "bicubic_direct_square",
                "align_corners": False,
                "antialias": True,
                "mean": list(self.loading['mean']),
                "std": list(self.loading['std']),
            },
            "open_clip_torch_version": self._open_clip_version,
            "cache_identity": self.cache_identity,
            "requires_true_local_tokens": self.require_true_local_tokens,
            "loading": self.loading,
        }

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("代理模型输入必须具有 [批, 3, 高, 宽] 形状。")
        image = images.to(self.device, dtype=torch.float32)
        if image.shape[-2:] != (self.image_size, self.image_size):
            image = F.interpolate(
                image,
                size=(self.image_size, self.image_size),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
        return (image - self.mean) / self.std

    def encode_image(self, images: torch.Tensor) -> ImageFeatures:
        normalized = self.preprocess(images)
        visual = self.model.visual
        if not hasattr(visual, "forward_intermediates"):
            if self.require_true_local_tokens:
                raise RuntimeError(
                    f"代理 {self.config.name}/{self.config.pretrained} 没有局部特征接口，兼容实验已中止。"
                )
            global_embedding = self.model.encode_image(normalized, normalize=True)
            return ImageFeatures(global_embedding, global_embedding[:, None, :])

        output = visual.forward_intermediates(
            normalized,
            indices=[-1],
            output_fmt="NLC",
            output_extra_tokens=True,
        )
        if not isinstance(output, dict):
            raise RuntimeError("当前 open_clip_torch 返回了未知的局部特征结构。")
        try:
            global_embedding = F.normalize(output["image_features"], dim=-1)
            patch_tokens = F.normalize(output["image_intermediates"][-1], dim=-1)
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("代理模型局部特征接口缺少预期字段。") from exc
        if patch_tokens.ndim != 3:
            raise RuntimeError(f"代理模型局部特征形状异常：{tuple(patch_tokens.shape)}")
        if self.require_true_local_tokens and patch_tokens.shape[1] <= 1:
            raise RuntimeError("代理模型只返回了一个全局向量，不能冒充局部图块特征。")
        return ImageFeatures(global_embedding, patch_tokens)

    @torch.no_grad()
    def encode_text(self, text: str | list[str]) -> torch.Tensor:
        texts = [text] if isinstance(text, str) else text
        context = int(getattr(self.tokenizer, 'context_length', 77))
        if any(len(self.tokenizer.encode(t)) + 2 > context for t in texts):
            raise ValueError('文字目标超过模型容量，禁止静默截断')
        tokens = self.tokenizer(texts).to(self.device)
        return F.normalize(self.model.encode_text(tokens), dim=-1)

