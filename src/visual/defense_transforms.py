"""图像防御的真实前向与显式可微代理。

代码迁移自 dual_defense/src/visual/defense_transforms.py，
作为 PAPO (Preprocessing-Aware Pixel Optimization，预处理感知像素优化) 的图像防御变换与可微近似核心实现。

张量 JPEG 使用正确的八位 YCbCr、正交 DCT、分通道量化表和色度抽样，
仍不声称逐像素复现 libjpeg 的整数变换和色度插值。正式评估应调用
``pillow_jpeg_tensor``；自适应优化可设置 ``exact_forward=True``，使前向
与真实编解码完全一致，仅在反向使用张量代理的雅可比矩阵。
"""
from __future__ import annotations
from src.paths import relocate_io_path

import io
import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter


LUMINANCE_QUANT_TABLE = torch.tensor([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
], dtype=torch.float64)

CHROMINANCE_QUANT_TABLE = torch.tensor([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
], dtype=torch.float64)


def _validate_image(image):
    if image.ndim != 4 or image.shape[1] != 3 or min(image.shape[-2:]) < 1:
        raise ValueError("图片必须具有非空 [批次, 3, 高, 宽] 形状")
    if not image.is_floating_point():
        raise TypeError("图片必须是归一化浮点张量")


def _validate_jpeg(quality, subsampling):
    if isinstance(quality, bool) or int(quality) != quality or not 1 <= quality <= 100:
        raise ValueError("JPEG 质量必须是 [1, 100] 内的整数")
    if isinstance(subsampling, bool) or subsampling not in (0, 1, 2):
        raise ValueError("色度抽样必须是 0（4:4:4）、1（4:2:2）或 2（4:2:0）")


def get_jpeg_quant_matrix(quality, device, dtype, *, chrominance=False):
    _validate_jpeg(quality, 0)
    q = int(quality)
    scale = 5000 // q if q < 50 else 200 - 2 * q
    table = CHROMINANCE_QUANT_TABLE if chrominance else LUMINANCE_QUANT_TABLE
    return ((table * scale + 50) / 100).floor().clamp(1, 255).to(device=device, dtype=dtype)


def _round_ste(value):
    return value + (value.round() - value).detach()


def _jpeg_round_ste(value):
    # JPEG 的系数量化在绝对值域四舍五入，避免正好半整数时的偶数舍入。
    rounded = value.sign() * (value.abs() + 0.5).floor()
    return value + (rounded - value).detach()


def _block_codec(channel, table):
    batch, channels, height, width = channel.shape
    pad_h, pad_w = (-height) % 8, (-width) % 8
    padded = F.pad(channel, (0, pad_w, 0, pad_h), mode="replicate")
    ph, pw = padded.shape[-2:]
    blocks = padded.reshape(batch, channels, ph // 8, 8, pw // 8, 8)
    blocks = blocks.permute(0, 1, 2, 4, 3, 5)
    positions = torch.arange(8, dtype=channel.dtype, device=channel.device)
    frequencies = positions[:, None]
    basis = ((math.pi / 8) * frequencies * (positions[None, :] + 0.5)).cos() * 0.5
    basis = torch.cat((basis[:1] / math.sqrt(2), basis[1:]), dim=0)
    coefficients = basis @ (blocks - 128.0) @ basis.T
    quantized = _jpeg_round_ste(coefficients / table) * table
    restored = basis.T @ quantized @ basis + 128.0
    restored = restored.permute(0, 1, 2, 4, 3, 5).reshape(batch, channels, ph, pw)
    return restored[:, :, :height, :width]


def tensor_jpeg_ste(image, quality=25, *, subsampling=2):
    """正确尺度的 JPEG 张量代理；色度上采样采用双线性近似。"""
    _validate_image(image)
    _validate_jpeg(quality, subsampling)
    original_dtype = image.dtype
    work = image if image.dtype in (torch.float32, torch.float64) else image.float()
    rgb = _round_ste(work.clamp(0, 1) * 255)
    red, green, blue = rgb.split(1, dim=1)
    y = 0.299 * red + 0.587 * green + 0.114 * blue
    cb = -0.168736 * red - 0.331264 * green + 0.5 * blue + 128
    cr = 0.5 * red - 0.418688 * green - 0.081312 * blue + 128
    y = _round_ste(y.clamp(0, 255))
    chroma = _round_ste(torch.cat((cb, cr), dim=1).clamp(0, 255))
    height, width = image.shape[-2:]
    factors = {0: (1, 1), 1: (1, 2), 2: (2, 2)}[subsampling]
    if subsampling:
        pad_h, pad_w = (-height) % factors[0], (-width) % factors[1]
        chroma = F.pad(chroma, (0, pad_w, 0, pad_h), mode="replicate")
        chroma = _round_ste(F.avg_pool2d(chroma, factors, stride=factors))
    luma_table = get_jpeg_quant_matrix(quality, work.device, work.dtype)
    chroma_table = get_jpeg_quant_matrix(quality, work.device, work.dtype, chrominance=True)
    y = _block_codec(y, luma_table)
    chroma = _block_codec(chroma, chroma_table)
    if subsampling:
        chroma = F.interpolate(chroma, scale_factor=factors, mode="bilinear", align_corners=False)
        chroma = chroma[:, :, :height, :width]
    cb, cr = (chroma - 128).split(1, dim=1)
    red = y + 1.402 * cr
    green = y - 0.344136 * cb - 0.714136 * cr
    blue = y + 1.772 * cb
    decoded = _round_ste(torch.cat((red, green, blue), dim=1).clamp(0, 255)) / 255
    return decoded.to(dtype=original_dtype)


def _pillow_batch(image, transform):
    _validate_image(image)
    levels = image.detach().clamp(0, 1).mul(255).round().to(device="cpu", dtype=torch.uint8)
    decoded = []
    for item in levels:
        array = item.permute(1, 2, 0).numpy()
        result = np.array(transform(Image.fromarray(array)), dtype=np.float32, copy=True) / 255.0
        decoded.append(torch.from_numpy(result).permute(2, 0, 1))
    return torch.stack(decoded).to(device=image.device, dtype=image.dtype)


def pillow_jpeg_tensor(image, quality=25, *, subsampling=2):
    """真实 Pillow JPEG 编解码；返回与规范八位重读一致的无梯度张量。"""
    _validate_jpeg(quality, subsampling)

    def transform(pil):
        buffer = io.BytesIO()
        pil.save(buffer, format="JPEG", quality=int(quality), subsampling=int(subsampling))
        buffer.seek(0)
        with Image.open(relocate_io_path(buffer)) as decoded:
            return decoded.convert("RGB")

    return _pillow_batch(image, transform)


def differentiable_jpeg_ste(image, quality, *, subsampling=2, exact_forward=False):
    """JPEG 的量化直通代理；可选择真实 Pillow 前向及代理反向。"""
    proxy = tensor_jpeg_ste(image, quality, subsampling=subsampling)
    if not exact_forward:
        return proxy
    actual = pillow_jpeg_tensor(image, quality, subsampling=subsampling)
    return actual + (proxy - proxy.detach())


def get_gaussian_kernel2d(sigma, device, dtype):
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("高斯标准差必须为有限正数")
    radius = int(math.ceil(3 * sigma))
    positions = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    kernel = (-0.5 * (positions / sigma).square()).exp()
    kernel = kernel / kernel.sum()
    return (kernel[:, None] * kernel[None, :]).expand(3, 1, -1, -1), radius


def differentiable_gaussian_blur(image, sigma):
    """可微高斯卷积；只作为 Pillow 扩展盒式滤波的反向代理。"""
    _validate_image(image)
    kernel, pad = get_gaussian_kernel2d(sigma, image.device, image.dtype)
    mode = "reflect" if pad < min(image.shape[-2:]) else "replicate"
    return F.conv2d(F.pad(image, (pad, pad, pad, pad), mode=mode), kernel, groups=3)


def pillow_gaussian_blur_tensor(image, radius=4.0):
    """真实 Pillow 高斯滤波；radius 按其定义即标准差，无额外校准。"""
    if not math.isfinite(radius) or radius < 0:
        raise ValueError("模糊半径必须为有限非负数")
    return _pillow_batch(image, lambda pil: pil.filter(ImageFilter.GaussianBlur(radius=radius)))


def gaussian_blur_bpda(image, radius=4.0, *, surrogate_sigma=None):
    """真实 Pillow 高斯前向与显式高斯卷积反向。"""
    actual = pillow_gaussian_blur_tensor(image, radius)
    sigma = radius if surrogate_sigma is None else surrogate_sigma
    proxy = image if sigma == 0 else differentiable_gaussian_blur(image, sigma)
    return actual + (proxy - proxy.detach())
