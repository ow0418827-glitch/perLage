"""新增的内容掩码指令生成器；原编辑器及其全局函数保持冻结。"""
from src.paths import relocate_io_path
from dataclasses import asdict
import hashlib, json, time
from src.paths import ProjectPath as Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from src.vision.natural_visual_edit import NaturalInstructionEditor, preserve_source, score_tensors, robust_score_tensors
from src.vision.diffusion_components import guided_noise_prediction
from src.vision.image_utils import read_tensor, save_tensor_image, write_json, sha256_file

class SurfaceInstructionEditor(NaturalInstructionEditor):

    def edit(self, source, target, output, plan, *, seed, target_guidance, title_feature, query_features, image_size=384, steps=30, strength=0.65, text_guidance=7.5, image_guidance=1.5, target_scale=8.0, latent_clip=0.04, source_weight=1.0, robust_guidance=False, edit_mask_path=None):
        """完整生成候选落盘；相同种子的无目标对照共享初始噪声与指令。"""
        if not isinstance(steps, int) or steps < 2 or image_size < 64 or image_size % 8 or (not 0 < strength <= 1):
            raise ValueError('无效编辑日程或尺寸')
        if edit_mask_path is None:
            raise ValueError('自然表面编辑需要显式内容保护掩码')
        edit_mask_path = Path(edit_mask_path).resolve()
        output = Path(output)
        identity = dict(source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(), target_sha256=hashlib.sha256(Path(target).read_bytes()).hexdigest(), plan=asdict(plan), mask_sha256=sha256_file(edit_mask_path), seed=seed, target_guidance=target_guidance, image_size=image_size, steps=steps, strength=strength, text_guidance=text_guidance, image_guidance=image_guidance, target_scale=target_scale, latent_clip=latent_clip, source_weight=source_weight, robust_guidance=robust_guidance, public_title=np.asarray(title_feature).tolist(), queries=np.asarray(query_features).tolist(), implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        identity = json.loads(json.dumps(identity))
        if output.exists() and any(output.iterdir()):
            prior_identity = output / 'identity.json'
            if not prior_identity.is_file() or json.loads(prior_identity.read_text(encoding='utf-8')) != identity:
                raise ValueError('候选目录非空且身份不符，拒绝覆盖')
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / 'identity.json', identity)
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        source_tensor = read_tensor(source, self.device)
        with Image.open(relocate_io_path(source)) as im, Image.open(relocate_io_path(edit_mask_path)) as raw:
            if im.size != raw.size:
                raise ValueError('内容保护掩码与源图尺寸不同')
            mask_image = raw.convert('L').copy()
        mask_image.save(output / 'edit_mask.png')
        mask = torch.from_numpy(np.asarray(mask_image).copy()).to(self.device).float()[None, None] / 255
        h, w = source_tensor.shape[-2:]
        shape = tuple((max(64, int(round(v * image_size / max(h, w) / 8)) * 8) for v in (h, w)))
        work = F.interpolate(source_tensor, size=shape, mode='bicubic', align_corners=False, antialias=True).clamp(0, 1)
        with torch.no_grad():
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
        latent_mask = F.interpolate(mask, size=latent.shape[-2:], mode='bilinear', align_corners=False)
        conditions = torch.cat([image_condition, image_condition, torch.zeros_like(image_condition)])
        title = torch.as_tensor(title_feature, device=self.device).float()[None]
        queries = torch.as_tensor(query_features, device=self.device).float()
        trace = []
        resumed_seconds = 0.0
        checkpoint = output / 'checkpoint.pt'
        start_index = 0
        if checkpoint.is_file():
            saved = torch.load(relocate_io_path(checkpoint), map_location=self.device, weights_only=True)
            if saved['identity'] != identity:
                raise ValueError('恢复检查点身份不符')
            latent, trace = (saved['latent'], saved['trace'])
            start_index = int(saved['completed_steps'])
            resumed_seconds = float(saved['elapsed_seconds'])
        for index, timestep in enumerate(times):
            if index < start_index:
                continue
            with torch.no_grad():
                inputs = self.scheduler.scale_model_input(latent, timestep)
                inputs = torch.cat([inputs] * 3)
                inputs = torch.cat([inputs, conditions], dim=1)
                eps = self.unet(inputs.to(self.unet.dtype), timestep, encoder_hidden_states=emb).sample.float()
                eps_text, eps_image, eps_uncond = eps.chunk(3)
                eps = eps_uncond + text_guidance * (eps_text - eps_image) + image_guidance * (eps_image - eps_uncond)
            t = int(timestep)
            alpha = self.scheduler.alphas_cumprod[t].to(self.device)
            next_t = int(times[index + 1]) if index + 1 < len(times) else -1
            alpha_previous = self.scheduler.alphas_cumprod[next_t].to(self.device) if next_t >= 0 else alpha.new_tensor(1.0)
            record = {'step': index + 1, 'timestep': t, 'guided': bool(target_guidance)}
            if target_guidance:
                variable = latent.detach().requires_grad_(True)
                clean_estimate = (variable - (1 - alpha).sqrt() * eps) / alpha.sqrt()
                decoded = (self.vae.decode(clean_estimate / self.vae.config.scaling_factor).sample / 2 + 0.5).clamp(0, 1)
                candidate = preserve_source(decoded, source_tensor, mask)
                if robust_guidance:
                    from src.vision.image_utils import transformed_batch
                    transformed = transformed_batch(candidate, step=index)
                    feature = self.encoder.surrogate.encode_image(transformed).global_embedding
                    score, image_score, joint_score, per_condition = robust_score_tensors(feature, target_feature, title, queries)
                    record['condition_image_cosines'] = per_condition.detach().cpu().tolist()
                else:
                    feature = self.encoder.surrogate.encode_image(candidate).global_embedding
                    score, image_score, joint_score = score_tensors(feature, target_feature, title, queries)
                preservation = ((candidate - source_tensor).square() * mask).mean()
                utility = score - source_weight * preservation
                gradient, = torch.autograd.grad(utility, variable)
                gradient = gradient * latent_mask
                eps, audit = guided_noise_prediction(eps, gradient, alpha, alpha_previous, guidance_scale=target_scale, latent_step_clip=latent_clip)
                record.update(audit)
                record.update(image_score=float(image_score.detach()), joint_score=float(joint_score.detach()), preservation=float(preservation.detach()))
                del variable, clean_estimate, decoded, candidate, feature, score, utility, gradient
            with torch.no_grad():
                latent = self.scheduler.step(eps, timestep, latent, eta=0.0).prev_sample.float()
                reference = self.scheduler.add_noise(source_latent, noise, times[index + 1:index + 2]) if next_t >= 0 else source_latent
                latent = latent * latent_mask + reference * (1 - latent_mask)
            if not torch.isfinite(latent).all():
                raise FloatingPointError('潜变量包含非有限数')
            trace.append(record)
            pending = output / 'checkpoint.pt.tmp'
            torch.save({'identity': identity, 'latent': latent.detach().cpu(), 'trace': trace, 'completed_steps': index + 1, 'elapsed_seconds': resumed_seconds + time.perf_counter() - started}, relocate_io_path(pending))
            pending.replace(checkpoint)
            write_json(output / 'progress.json', {'status': 'running', 'seed': seed, 'completed_steps': index + 1, 'total_steps': len(times), 'last': record})
        with torch.no_grad():
            decoded = (self.vae.decode(latent / self.vae.config.scaling_factor).sample / 2 + 0.5).clamp(0, 1)
            final = preserve_source(decoded, source_tensor, mask)
        save_tensor_image(output / 'edited.png', final)
        reread = read_tensor(output / 'edited.png', self.device)
        protected = mask == 0
        protected_delta = (reread - source_tensor).abs().masked_select(protected.expand_as(source_tensor))
        if protected_delta.numel() and float(protected_delta.max()) > 0:
            raise AssertionError('受保护区域的源像素发生变化')
        metadata = dict(plan=asdict(plan), seed=seed, target_guidance=target_guidance, steps=steps, completed_steps=len(times), strength=strength, working_shape=shape, text_guidance=text_guidance, image_guidance=image_guidance, target_scale=target_scale, robust_guidance=robust_guidance, latent_clip=latent_clip, source_weight=source_weight, editable_fraction=float(mask.mean()), protected_max_difference=0, generation_seconds=resumed_seconds + time.perf_counter() - started, peak_allocated_bytes=torch.cuda.max_memory_allocated() if self.device.type == 'cuda' else None, planner='explicit_reviewed_recipe_not_language_model', editor='content_mask_instruction_conditioned_latent_diffusion', verifier='ink_core_pixel_preservation_plus_manual_content_and_logo_review', original_pixel_budget_enforced=False, naturalness_certified=False, base_model=str(self.base_model.resolve()), editor_model=str(self.editor_model.resolve()), implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), trace=trace)
        write_json(output / 'metadata.json', metadata)
        write_json(output / 'progress.json', {'status': 'completed', 'completed_steps': len(times)})
        return (output / 'edited.png', metadata)
