"""完整方法的生成配置、初始化锚点与质量检查。"""
from __future__ import annotations

from src.model_identity import model_identity_matches

from src.paths import relocate_io_path

from dataclasses import asdict, dataclass

import gc

import hashlib

import json

from src.paths import ProjectPath as Path

import time

import numpy as np

import torch

from src.vision._natural.core import NaturalInitializationConfig, source_ssim

from src.vision.natural_visual_edit import EditPlan, NaturalInstructionEditor

from src.vision.image_utils import read_tensor, quantize_ste, save_tensor_image, sha256_file, write_json

from src.vision.retrieval_objective import TRAINING_DATA_POLICY, training_data

@dataclass(frozen=True)
class NaturalGenerationConfig:
    seed: int = 42
    target_scale: float = 8.0
    latent_clip: float = 0.04
    strength: float = 1.0
    steps: int = 30
    image_size: int = 384
    source_weight: float = 1.0
    robust_guidance: bool = True

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def prepare_generation_anchor(source, generated, output, *, device):
    """只按固定源图质量约束取最大混合量；不比较候选排名。"""
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    original, edited = (read_tensor(source, device), read_tensor(generated, device))
    if original.shape != edited.shape:
        raise ValueError('生成图必须保留源图尺寸')
    trials = []
    chosen = None
    with torch.no_grad():
        for index in range(32, 3, -1):
            alpha = index / 32
            image = quantize_ste((original + alpha * (edited - original)).clamp(0, 1))
            quality = float(source_ssim(image, original))
            mae = float((image - original).abs().mean() * 255)
            trials.append(dict(alpha=alpha, ssim=quality, mae=mae))
            if quality >= 0.97001 and mae <= 4.0:
                chosen = image
                break
    record = dict(source_sha256=sha256_file(source), generated_sha256=sha256_file(generated), trials=trials, selection='largest_fixed_one_over_32_blend_with_ssim_at_least_0p97_mae_at_most_4', private_gallery_used=False, true_rank_used=False)
    if chosen is None:
        write_json(root / 'quality_rejection.json', record)
        raise ValueError('自然初始化未通过固定质量门槛')
    image = root / 'generative_anchor.png'
    save_tensor_image(image, chosen)
    record.update(chosen=trials[-1], prepared_sha256=sha256_file(image), status='completed')
    identity_path = root / 'identity.json'
    if identity_path.exists() and read_json(identity_path) != record:
        raise ValueError('自然初始化身份改变')
    write_json(identity_path, record)
    return image
