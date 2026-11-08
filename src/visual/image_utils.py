"""供保留方法共用的基础工具；不提供其他图片优化方法。"""
from __future__ import annotations
from src.paths import relocate_io_path

import hashlib
import json
from src.paths import ProjectPath as Path
from typing import Any
import numpy as np
from PIL import Image
import torch
from src.visual.defense_transforms import gaussian_blur_bpda, differentiable_jpeg_ste

def sha256_file(path: str | Path | None) -> str | None:
    if path is None or hasattr(path, "convert"):
        return None
    p = Path(path)
    if not p.is_file():
        return None
    hasher = hashlib.sha256()
    with p.open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def read_tensor(path: str | Path, device: torch.device | str) -> torch.Tensor:
    with Image.open(relocate_io_path(path)) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32) / 255.0


def save_tensor_image(path: str | Path, image: torch.Tensor) -> np.ndarray:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = image.detach()[0].mul(255).round().clamp(0, 255).permute(1, 2, 0).byte().cpu().numpy()
    Image.fromarray(array).save(path, format="PNG")
    return array


def quantize_ste(image: torch.Tensor) -> torch.Tensor:
    """八位取整直通估计（STE: Straight-Through Estimator）：真实量化前向、恒等反向。"""
    rounded = image.clamp(0, 1).mul(255).round().div(255)
    return image + (rounded - image).detach()


def project_pixels(source: torch.Tensor, delta: torch.Tensor, epsilon: float) -> torch.Tensor:
    """联合像素投影：相对源图总 L_infinity 扰动约束与合法像素范围 [0, 1] 联合裁剪。"""
    lower = torch.maximum(-source, source.new_tensor(-epsilon))
    upper = torch.minimum(1.0 - source, source.new_tensor(epsilon))
    return torch.maximum(torch.minimum(delta, upper), lower)


def transformed_batch(image: torch.Tensor, *, step: int | None = None) -> torch.Tensor:
    """三分支防御预处理批次。

    训练时按步数固定配对循环：
      半径 (3, 4, 5) 与 JPEG 质量 (20, 25, 35)；JPEG 采用 4:2:0 色度抽样 (subsampling=2)。
    定期检查候选时使用固定评测条件：半径 4.0、质量 25。
    """
    radius = 4.0 if step is None else (3.0, 4.0, 5.0)[(step - 1) % 3]
    quality = 25 if step is None else (20, 25, 35)[(step - 1) % 3]
    quantized = quantize_ste(image)
    blur = gaussian_blur_bpda(quantized, radius)
    jpeg = differentiable_jpeg_ste(quantized, quality, subsampling=2, exact_forward=True)
    return torch.cat([quantized, blur, jpeg], dim=0)
