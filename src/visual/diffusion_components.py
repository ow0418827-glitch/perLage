"""供保留方法共用的基础工具；不提供其他图片优化方法。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from src.paths import ProjectPath as Path
from typing import Any

@dataclass
class DiffusionComponents:
    """支持真实组件和仅用于单元验证的微型组件注入。"""
    unet: Any
    vae: Any
    scheduler: Any
    inverse_scheduler: Any
    text_embeddings: Any
    manifest: dict[str, Any]
    noise_forward_calls: int = 0


def _model_manifest(model_root):
    """记录本地权重与配置内容，避免只凭目录名宣称模型身份。"""
    files = {}
    for subfolder in ("unet", "vae", "text_encoder", "tokenizer", "scheduler"):
        for path in sorted((model_root / subfolder).glob("*")):
            if path.is_file() and path.suffix in {".json", ".txt", ".safetensors", ".bin"}:
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                files[str(path.relative_to(model_root)).replace("\\", "/")] = {
                    "sha256": digest.hexdigest(), "bytes": path.stat().st_size,
                }
    return files


def load_diffusion_components(settings, device):
    """仅加载本地必需组件；图像编码解码使用单精度，噪声网络用半精度。"""
    import torch
    from diffusers import AutoencoderKL, DDIMInverseScheduler, DDIMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    model_root = Path(settings["model_path"])
    if not model_root.is_dir():
        raise FileNotFoundError(f"本地扩散模型目录不存在：{model_root}")
    network_dtype = torch.float16 if device.type == "cuda" else torch.float32
    common = {"local_files_only": True, "use_safetensors": True}
    if settings["variant"]:
        common["variant"] = settings["variant"]
    unet = UNet2DConditionModel.from_pretrained(
        model_root, subfolder="unet", torch_dtype=network_dtype, **common)
    vae = AutoencoderKL.from_pretrained(
        model_root, subfolder="vae", torch_dtype=torch.float32, **common)
    text_encoder = CLIPTextModel.from_pretrained(
        model_root, subfolder="text_encoder", torch_dtype=network_dtype, **common)
    tokenizer = CLIPTokenizer.from_pretrained(
        model_root, subfolder="tokenizer", local_files_only=True)
    scheduler = DDIMScheduler.from_pretrained(
        model_root, subfolder="scheduler", local_files_only=True,
        clip_sample=False, set_alpha_to_one=True, steps_offset=0, timestep_spacing="leading")
    if scheduler.config.prediction_type != "epsilon":
        raise ValueError("当前扩散引导仅支持 epsilon 噪声预测模型")
    inverse = DDIMInverseScheduler.from_config(scheduler.config)
    for model in (unet, vae, text_encoder):
        model.eval().requires_grad_(False).to(device)
    if settings["attention_slicing"]:
        unet.set_attention_slice("auto")
    prompts = [settings["prompt"]]
    if settings["prompt_guidance_scale"] > 1:
        prompts.insert(0, "")
    tokenized = tokenizer(prompts, padding="max_length", max_length=tokenizer.model_max_length,
                          truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings = text_encoder(tokenized.input_ids.to(device))[0].detach()
    text_encoder.to("cpu")
    del text_encoder, tokenizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return DiffusionComponents(unet, vae, scheduler, inverse, embeddings,
        {"local_path": str(model_root), "local_files_only": True,
         "variant": settings["variant"], "files": _model_manifest(model_root)})


def guided_noise_prediction(noise_prediction, gradient, alpha, alpha_previous, *,
                            guidance_scale, latent_step_clip, normalize_gradient=True):
    """最大化目标的得分引导，并限制其带来的实际单步潜变量位移。

    DDIM 的上一时刻样本中，噪声预测的系数为
    sqrt(1-a_prev)-sqrt(a_prev)*sqrt((1-a)/a)，在去噪方向小于零。
    所以 epsilon_hat = epsilon - scale*sqrt(1-a)*grad(J) 会沿 +grad(J)
    方向移动。这里 J 是相似度减重建惩罚，不是待最小化的损失。
    """
    import torch
    if not torch.isfinite(gradient).all():
        raise FloatingPointError("扩散引导梯度包含非有限数值")
    rms = gradient.square().mean().sqrt()
    if float(rms) <= 1e-12 or guidance_scale == 0:
        return noise_prediction, {"gradient_rms": float(rms), "guidance_displacement_linf": 0.0,
                                  "guidance_clip_factor": 1.0, "zero_gradient": float(rms) <= 1e-12}
    direction = gradient / rms.clamp_min(1e-12) if normalize_gradient else gradient
    correction = -float(guidance_scale) * (1 - alpha).sqrt() * direction
    coefficient = (1 - alpha_previous).sqrt() - alpha_previous.sqrt() * ((1 - alpha) / alpha).sqrt()
    displacement_max = float((coefficient * correction).abs().max())
    clip_factor = min(1.0, float(latent_step_clip) / max(displacement_max, 1e-12))
    correction = correction * clip_factor
    result = noise_prediction + correction
    if not torch.isfinite(result).all():
        raise FloatingPointError("引导后的噪声预测包含非有限数值")
    return result.detach(), {"gradient_rms": float(rms),
        "guidance_displacement_linf": displacement_max * clip_factor,
        "guidance_clip_factor": clip_factor, "zero_gradient": False}
