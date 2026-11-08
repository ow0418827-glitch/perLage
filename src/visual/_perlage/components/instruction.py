"""独立的规划—指令编辑—可审计验证适配；不修改旧攻击路由。

真实使用轻量指令条件扩散模型。规划是显式配方，验证不是语言模型
判断，也不宣称严格自然流形。目标梯度只更新扩散潜变量，最后再次
保留受保护区域的源像素。没有原方法的全图像素预算。
"""
from __future__ import annotations
from src.paths import relocate_io_path

from dataclasses import asdict, dataclass
import hashlib
import json
from src.paths import ProjectPath as Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch
import torch.nn.functional as F

from src.visual.diffusion_components import guided_noise_prediction
from src.visual.image_utils import read_tensor, save_tensor_image, write_json


@dataclass(frozen=True)
class EditPlan:
    instruction: str
    editable_regions: tuple[tuple[float, float, float, float], ...]
    reason: str
    region_policy: str = "protected_source"
    feather_pixels: float = 3.0
    protected_regions: tuple[tuple[float, float, float, float], ...] = ()

    def __post_init__(self):
        if not self.instruction.strip() or not self.editable_regions:
            raise ValueError("编辑指令和区域不能为空")
        if self.region_policy not in {"protected_source", "whole_image_diagnostic", "label_protected"}:
            raise ValueError("未知区域策略")
        if not 0 <= self.feather_pixels <= 32:
            raise ValueError("羽化宽度无效")
        for x0, y0, x1, y1 in (*self.editable_regions, *self.protected_regions):
            if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
                raise ValueError("编辑区域必须是归一化有效矩形")


def edit_mask(size, plan: EditPlan):
    """掩码为一的区域允许编辑；羽化仅向编辑区域内部收缩。"""
    width, height = size
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for x0, y0, x1, y1 in plan.editable_regions:
        draw.rectangle((round(x0 * width), round(y0 * height),
                        min(width - 1, round(x1 * width) - 1),
                        min(height - 1, round(y1 * height) - 1)), fill=255)
    for x0, y0, x1, y1 in plan.protected_regions:
        draw.rectangle((round(x0 * width), round(y0 * height),
                        min(width - 1, round(x1 * width) - 1),
                        min(height - 1, round(y1 * height) - 1)), fill=0)
    hard = np.asarray(mask).copy()
    if plan.feather_pixels:
        soft = np.asarray(mask.filter(ImageFilter.GaussianBlur(plan.feather_pixels)))
        mask = Image.fromarray(np.minimum(hard, soft))
    return mask


def preserve_source(edited, original, mask):
    """输出尺寸以源图为准，掩码零区域精确保留源像素。"""
    if edited.ndim != 4 or original.ndim != 4 or original.shape[1] != 3:
        raise ValueError("图片必须是四维三通道张量")
    if mask.ndim != 4 or mask.shape[:2] != (1, 1):
        raise ValueError("掩码必须具有单批次单通道")
    edited = F.interpolate(edited, size=original.shape[-2:], mode="bicubic",
                           align_corners=False, antialias=True).clamp(0, 1)
    if mask.shape[-2:] != original.shape[-2:]:
        raise ValueError("源图和最终掩码尺寸不一致")
    if not torch.isfinite(mask).all() or not ((mask >= 0) & (mask <= 1)).all():
        raise ValueError("掩码必须是零至一之间的有限数")
    return original * (1 - mask) + edited * mask


def score_tensors(feature, target, history, queries):
    """已知图像与三条开发表述的联合对齐；最终表述不进入此目标。"""
    image_score = (feature * target).sum(-1).mean()
    joint = F.normalize(feature + history, dim=-1)
    similarities = (queries * joint).sum(-1)
    joint_score = -0.04 * torch.logsumexp(-similarities / 0.04, dim=0)
    return image_score + 0.5 * joint_score, image_score, similarities.mean()


def robust_score_tensors(features, target, history, queries):
    """每个真实预处理条件覆盖全部开发表述，再优化最弱条件。"""
    image_scores = (features * target).sum(-1)
    joint = F.normalize(features + history, dim=-1)
    similarities = joint @ queries.T
    joint_scores = -0.04 * torch.logsumexp(-similarities / 0.04, dim=-1)
    utilities = image_scores + 0.5 * joint_scores
    value = -0.04 * torch.logsumexp(-utilities / 0.04, dim=0)
    return value, image_scores.min(), similarities.mean(), image_scores


class NaturalInstructionEditor:
    """复用基础组件，仅另载指令去噪器；全部模型权重冻结。"""

    def __init__(self, base_model, editor_model, encoder, device="cuda"):
        from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTokenizer

        self.device = torch.device(device)
        self.encoder = encoder
        self.base_model, self.editor_model = Path(base_model), Path(editor_model)
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        options = dict(local_files_only=True, use_safetensors=True, variant="fp16")
        self.unet = UNet2DConditionModel.from_pretrained(
            editor_model, subfolder="unet", torch_dtype=dtype, **options)
        self.vae = AutoencoderKL.from_pretrained(
            base_model, subfolder="vae", torch_dtype=torch.float32, **options)
        self.text_encoder = CLIPTextModel.from_pretrained(
            base_model, subfolder="text_encoder", torch_dtype=dtype, **options)
        self.tokenizer = CLIPTokenizer.from_pretrained(
            base_model, subfolder="tokenizer", local_files_only=True)
        self.scheduler = DDIMScheduler.from_pretrained(
            base_model, subfolder="scheduler", local_files_only=True,
            clip_sample=False, set_alpha_to_one=True, steps_offset=0,
            timestep_spacing="leading")
        if self.unet.config.in_channels != 8:
            raise ValueError("指令编辑去噪器必须是八通道输入")
        for model in (self.unet, self.vae):
            model.eval().requires_grad_(False).to(self.device)
        self.text_encoder.eval().requires_grad_(False).to("cpu")
        self.unet.set_attention_slice("auto")
        self._embeddings = {}

    def embeddings(self, prompt):
        if prompt not in self._embeddings:
            self.text_encoder.to(self.device)
            tokens = self.tokenizer([prompt, "", ""], padding="max_length",
                                    max_length=77, truncation=True, return_tensors="pt")
            with torch.no_grad():
                self._embeddings[prompt] = self.text_encoder(tokens.input_ids.to(self.device))[0].detach()
            self.text_encoder.to("cpu")
            torch.cuda.empty_cache()
        return self._embeddings[prompt]

    def edit(self, source, target, output, plan, *, seed, target_guidance,
             history_feature, query_features, image_size=384, steps=30,
             strength=0.65, text_guidance=7.5, image_guidance=1.5,
             target_scale=8.0, latent_clip=0.04, source_weight=1.0,
             robust_guidance=False):
        """完整生成候选落盘；相同种子的无目标对照共享初始噪声与指令。"""
        if (not isinstance(steps, int) or steps < 2 or image_size < 64
                or image_size % 8 or not 0 < strength <= 1):
            raise ValueError("无效编辑日程或尺寸")
        output = Path(output)
        identity = dict(source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),
            target_sha256=hashlib.sha256(Path(target).read_bytes()).hexdigest(),
            plan=asdict(plan), seed=seed, target_guidance=target_guidance,
            image_size=image_size, steps=steps, strength=strength,
            text_guidance=text_guidance, image_guidance=image_guidance,
            target_scale=target_scale, latent_clip=latent_clip, source_weight=source_weight,
            robust_guidance=robust_guidance,
            history=np.asarray(history_feature).tolist(), queries=np.asarray(query_features).tolist(),
            implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        identity = json.loads(json.dumps(identity))
        if output.exists() and any(output.iterdir()):
            prior_identity = output / "identity.json"
            if not prior_identity.is_file() or json.loads(prior_identity.read_text(encoding="utf-8")) != identity:
                raise ValueError("候选目录非空且身份不符，拒绝覆盖")
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "identity.json", identity)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        source_tensor = read_tensor(source, self.device)
        with Image.open(relocate_io_path(source)) as im:
            mask_image = edit_mask(im.size, plan)
        mask_image.save(output / "edit_mask.png")
        mask = torch.from_numpy(np.asarray(mask_image).copy()).to(self.device).float()[None, None] / 255
        h, w = source_tensor.shape[-2:]
        shape = tuple(max(64, int(round(v * image_size / max(h, w) / 8)) * 8) for v in (h, w))
        work = F.interpolate(source_tensor, size=shape, mode="bicubic", align_corners=False,
                             antialias=True).clamp(0, 1)
        with torch.no_grad():
            # 指令模型的源图条件不乘缩放因子，与官方准备图像潜变量逻辑一致。
            image_condition = self.vae.encode(work * 2 - 1).latent_dist.mode().float()
            source_latent = image_condition * self.vae.config.scaling_factor
            target_tensor = read_tensor(target, self.device)
            target_feature = self.encoder.surrogate.encode_image(target_tensor).global_embedding.detach()
        emb = self.embeddings(plan.instruction)
        self.scheduler.set_timesteps(steps, device=self.device)
        times = self.scheduler.timesteps[steps - max(1, int(steps * strength)):]
        generator = torch.Generator(device=self.device).manual_seed(seed)
        noise = torch.randn(source_latent.shape, generator=generator, device=self.device)
        latent = self.scheduler.add_noise(source_latent, noise, times[:1]).float()
        latent_mask = F.interpolate(mask, size=latent.shape[-2:], mode="bilinear", align_corners=False)
        conditions = torch.cat([image_condition, image_condition, torch.zeros_like(image_condition)])
        history = torch.as_tensor(history_feature, device=self.device).float()[None]
        queries = torch.as_tensor(query_features, device=self.device).float()
        trace = []
        resumed_seconds = 0.
        checkpoint = output / "checkpoint.pt"
        start_index = 0
        if checkpoint.is_file():
            saved = torch.load(relocate_io_path(checkpoint), map_location=self.device, weights_only=True)
            if saved["identity"] != identity:
                raise ValueError("恢复检查点身份不符")
            latent, trace = saved["latent"], saved["trace"]
            start_index = int(saved["completed_steps"])
            resumed_seconds = float(saved["elapsed_seconds"])
        for index, timestep in enumerate(times):
            if index < start_index:
                continue
            with torch.no_grad():
                inputs = self.scheduler.scale_model_input(latent, timestep)
                inputs = torch.cat([inputs] * 3)
                inputs = torch.cat([inputs, conditions], dim=1)
                eps = self.unet(inputs.to(self.unet.dtype), timestep,
                                encoder_hidden_states=emb).sample.float()
                eps_text, eps_image, eps_uncond = eps.chunk(3)
                eps = eps_uncond + text_guidance * (eps_text - eps_image) + image_guidance * (eps_image - eps_uncond)
            t = int(timestep)
            alpha = self.scheduler.alphas_cumprod[t].to(self.device)
            next_t = int(times[index + 1]) if index + 1 < len(times) else -1
            alpha_previous = self.scheduler.alphas_cumprod[next_t].to(self.device) if next_t >= 0 else alpha.new_tensor(1.0)
            record = {"step": index + 1, "timestep": t, "guided": bool(target_guidance)}
            if target_guidance:
                variable = latent.detach().requires_grad_(True)
                clean_estimate = (variable - (1 - alpha).sqrt() * eps) / alpha.sqrt()
                decoded = (self.vae.decode(clean_estimate / self.vae.config.scaling_factor).sample / 2 + 0.5).clamp(0, 1)
                candidate = preserve_source(decoded, source_tensor, mask)
                if robust_guidance:
                    from src.visual.image_utils import transformed_batch
                    transformed = transformed_batch(candidate, step=index)
                    feature = self.encoder.surrogate.encode_image(transformed).global_embedding
                    score, image_score, joint_score, per_condition = robust_score_tensors(
                        feature, target_feature, history, queries)
                    record["condition_image_cosines"] = per_condition.detach().cpu().tolist()
                else:
                    feature = self.encoder.surrogate.encode_image(candidate).global_embedding
                    score, image_score, joint_score = score_tensors(feature, target_feature, history, queries)
                preservation = ((candidate - source_tensor).square() * mask).mean()
                utility = score - source_weight * preservation
                gradient, = torch.autograd.grad(utility, variable)
                gradient = gradient * latent_mask
                eps, audit = guided_noise_prediction(eps, gradient, alpha, alpha_previous,
                    guidance_scale=target_scale, latent_step_clip=latent_clip)
                record.update(audit)
                record.update(image_score=float(image_score.detach()),
                              joint_score=float(joint_score.detach()),
                              preservation=float(preservation.detach()))
                del variable, clean_estimate, decoded, candidate, feature, score, utility, gradient
            with torch.no_grad():
                latent = self.scheduler.step(eps, timestep, latent, eta=0.0).prev_sample.float()
                reference = (self.scheduler.add_noise(source_latent, noise, times[index + 1:index + 2])
                             if next_t >= 0 else source_latent)
                latent = latent * latent_mask + reference * (1 - latent_mask)
            if not torch.isfinite(latent).all():
                raise FloatingPointError("潜变量包含非有限数")
            trace.append(record)
            pending = output / "checkpoint.pt.tmp"
            torch.save({"identity": identity, "latent": latent.detach().cpu(),
                "trace": trace, "completed_steps": index + 1,
                "elapsed_seconds": resumed_seconds + time.perf_counter() - started}, relocate_io_path(pending))
            pending.replace(checkpoint)
            write_json(output / "progress.json", {"status": "running", "seed": seed,
                "completed_steps": index + 1, "total_steps": len(times), "last": record})
        with torch.no_grad():
            decoded = (self.vae.decode(latent / self.vae.config.scaling_factor).sample / 2 + 0.5).clamp(0, 1)
            final = preserve_source(decoded, source_tensor, mask)
        save_tensor_image(output / "edited.png", final)
        reread = read_tensor(output / "edited.png", self.device)
        protected = mask == 0
        protected_delta = (reread - source_tensor).abs().masked_select(protected.expand_as(source_tensor))
        if protected_delta.numel() and float(protected_delta.max()) > 0:
            raise AssertionError("受保护区域的源像素发生变化")
        metadata = dict(plan=asdict(plan), seed=seed, target_guidance=target_guidance,
                        steps=steps, completed_steps=len(times), strength=strength,
                        working_shape=shape, text_guidance=text_guidance,
                        image_guidance=image_guidance, target_scale=target_scale,
                        robust_guidance=robust_guidance,
                        latent_clip=latent_clip, source_weight=source_weight,
                        editable_fraction=float(mask.mean()), protected_max_difference=0,
                        generation_seconds=resumed_seconds + time.perf_counter() - started,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated() if self.device.type == "cuda" else None,
                        planner="explicit_reviewed_recipe_not_language_model",
                        editor="instruction_conditioned_latent_diffusion",
                        verifier="pixel_preservation_and_retrieval_audit_plus_manual_visual_review",
                        original_pixel_budget_enforced=False,
                        naturalness_certified=False,
                        base_model=str(self.base_model.resolve()),
                        editor_model=str(self.editor_model.resolve()),
                        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        trace=trace)
        write_json(output / "metadata.json", metadata)
        write_json(output / "progress.json", {"status": "completed", "completed_steps": len(times)})
        return output / "edited.png", metadata
