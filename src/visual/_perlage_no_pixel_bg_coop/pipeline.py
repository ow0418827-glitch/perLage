"""独立主体锁定（同 PerLAGE_no_pixel）＋背景潜空间优化试验；原无像素方法保持只读。

当 latent_levels == 0 时，商品主体前景严格锁定为同组已完成 PerLAGE_no_pixel 的最终图像像素（零额外改动），
仅对主体以外的背景区域执行 350 步潜空间平滑优化。
训练输入仍仅限公开图对、源商品名、自身公开全文键及原冻结的一条目标特点正文。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import gc
import hashlib
import json
import math
import time

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from types import SimpleNamespace
import torch
import torch.nn.functional as F

from src.paths import PROJECT, ProjectPath as Path, model_path, relocate_io_path
from src.visual.image_utils import read_tensor, quantize_ste, save_tensor_image, sha256_file, write_json
from src.visual._perlage.components.colour import read_rgb, prepare_print_context, halo_error, print_region_mask
from src.visual._perlage.components.core import source_ssim, tangent_projection
from src.visual._perlage.components.evidence import STAGES, stage_record
from src.visual._perlage_no_pixel.evidence import validate_complete as validate_original_complete
from src.visual._perlage_no_pixel.pipeline import validate_definition
from src.visual.retrieval_objective import TRAINING_DATA_POLICY, training_data
from src.visual.robust_objectives import softmin, compensated_targets
from src.visual.target_features import format_retrieval_text
from src.visual.encoders.joint import ChunkedClipEncoder


METHOD = 'PerLAGE_no_pixel_bg_coop_experiment'
DEFINITION = 'frozen_perlage_no_pixel_subject_locked_background_latent_v2'
MASK_POLICY = 'subject_locked_to_perlage_no_pixel_background_outward_feathered_latent_v2'


@dataclass(frozen=True)
class BgCoopLatentConfig:
    steps: int = 350
    check_every: int = 10
    latent_levels: int = 0
    background_levels: int = 24
    latent_working_size: int = 512
    latent_limit: float = 6.0
    latent_learning_rate: float = 0.06
    latent_smoothing_mix: float = 0.0
    source_ssim_floor: float = 0.75
    source_mae_ceiling: float = 28.0
    temperature: float = 0.04
    macro_l2_weight: float = 0.0
    macro_tv_weight: float = 0.0
    bg_l2_weight: float = 0.0001
    bg_tv_weight: float = 0.0005
    overflow_weight: float = 0.05
    halo_soft_weight: float = 0.0001
    edge_feather_pixels: float = 2.0
    edge_feather_mode: str = 'linear'
    bg_luma_smooth_sigma: float = 0.0
    bg_chroma_smooth_sigma: float = 0.0
    bg_chroma_keep: float = 1.0
    bg_chroma_tv_weight: float = 0.0
    tau_scale: float = 1.0
    joint_weight: float = 0.5
    compensated_weight: float = 0.5
    halo_parent_multiplier: float = 1.0
    halo_floor: float = 12.0
    guard_backtracks: int = 8
    auxiliary_levels: int = 0
    feature_weight: float = 0.0
    diffusion_weight: float = 0.0
    seed: int = 42

    def validate(self):
        if type(self.steps) is not int or self.steps < 1 or type(self.check_every) is not int or self.check_every < 1 or self.steps % self.check_every:
            raise ValueError('试验须执行完整、可检查的细化步数')
        if type(self.latent_levels) is not int or not 0 <= self.latent_levels <= 128:
            raise ValueError('试验主体潜变量解码变化预算无效')
        if type(self.background_levels) is not int or not 1 <= self.background_levels <= 64:
            raise ValueError('试验背景潜空间优化预算无效')
        if not 0.25 <= self.source_ssim_floor < 1 or not 0 < self.source_mae_ceiling <= 128:
            raise ValueError('试验质量门槛无效')
        if not 0 < self.latent_limit <= 12 or not 0 < self.latent_learning_rate <= 0.2 or not 0 <= self.latent_smoothing_mix <= 1:
            raise ValueError('试验潜变量更新参数无效')
        if self.auxiliary_levels != 0 or self.feature_weight != 0 or self.diffusion_weight != 0:
            raise ValueError('本试验禁止像素残差、教师特征保持及扩散净化目标')
        if not 0 < self.temperature <= 1 or min(self.macro_l2_weight, self.macro_tv_weight, self.bg_l2_weight, self.bg_tv_weight, self.bg_chroma_tv_weight, self.overflow_weight) < 0:
            raise ValueError('试验目标或平滑正则参数无效')
        if self.halo_floor < 0 or self.halo_parent_multiplier < 0 or type(self.guard_backtracks) is not int or not 0 <= self.guard_backtracks <= 16:
            raise ValueError('试验自然度守门参数无效')
        if (self.halo_soft_weight != 0.0001 or not (2.0 <= self.edge_feather_pixels <= 32.0)
                or self.edge_feather_mode not in ('linear', 'cosine', 'exponential')
                or not (0.0 <= self.bg_luma_smooth_sigma <= 16.0)
                or not (0.0 <= self.bg_chroma_smooth_sigma <= 16.0)
                or not (0.1 <= self.bg_chroma_keep <= 1.0)
                or not (0.2 <= self.tau_scale <= 1.0)
                or self.joint_weight != 0.5 or self.compensated_weight != 0.5
                or self.latent_smoothing_mix != 0 or self.macro_l2_weight != 0):
            raise ValueError('试验须遵守已冻结目标权重及边界过渡方案')
        if type(self.seed) is not int or self.seed < 0 or self.latent_working_size < 64:
            raise ValueError('试验种子或编码尺寸无效')
        if any(not math.isfinite(v) for v in asdict(self).values() if isinstance(v, float)):
            raise ValueError('试验参数必须有限')


BG_COOP_PRESETS = {
    'fg_lock_bg24': asdict(BgCoopLatentConfig(
        steps=350, latent_levels=0, background_levels=24,
        source_ssim_floor=0.75, source_mae_ceiling=24.0,
        bg_l2_weight=0.0001, bg_tv_weight=0.0005, overflow_weight=0.05)),
    'fg_lock_bg36': asdict(BgCoopLatentConfig(
        steps=350, latent_levels=0, background_levels=36,
        source_ssim_floor=0.72, source_mae_ceiling=28.0,
        bg_l2_weight=0.0001, bg_tv_weight=0.0004, overflow_weight=0.05)),
    'bg_smooth': asdict(BgCoopLatentConfig(
        steps=350, latent_levels=0, background_levels=22,
        source_ssim_floor=0.78, source_mae_ceiling=22.0,
        edge_feather_pixels=12.0, edge_feather_mode='cosine',
        bg_luma_smooth_sigma=2.2, bg_chroma_smooth_sigma=5.0,
        bg_chroma_keep=0.55, tau_scale=0.75,
        bg_l2_weight=0.0001, bg_tv_weight=0.0015, bg_chroma_tv_weight=0.003,
        overflow_weight=0.05)),
}
PRESETS = BG_COOP_PRESETS


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _canonical(value):
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))


def build_source_public_key(record, source_title):
    """仅从调用方冻结的自身公开资料重建正式检索键，不接收记忆条目。"""
    allowed = {'id', 'title', 'brand', 'attributes', 'feature_text', 'provenance', 'feature_evidence'}
    if not isinstance(record, dict) or set(record) - allowed:
        raise ValueError('自身公开资料字段不在允许范围内')
    if record.get('title') != source_title or not record.get('id'):
        raise ValueError('自身公开资料与本组源商品名不一致')
    if not isinstance(record.get('feature_text'), str) or not record['feature_text'].strip():
        raise ValueError('自身公开资料缺少真实特点正文')
    if (not isinstance(record.get('brand', ''), str)
            or not isinstance(record.get('attributes'), (list, tuple))
            or any(not isinstance(value, str) or not value.strip() for value in record['attributes'])):
        raise ValueError('自身公开品牌或属性格式错误')
    text = format_retrieval_text(source_title, record.get('brand', ''), record['attributes'])
    return text + '\nFeatureText: ' + record['feature_text'].strip()


class PublicFullKeyObjective:
    """完整自身公开文字键与图像各半，查询仍为原目标图及唯一训练句各半。"""
    def __init__(self, surrogate, target_feature, source_public_key, training_queries):
        if len(training_queries) != 1 or not isinstance(training_queries[0], str) or not training_queries[0].strip():
            raise ValueError('本轮严格复用一条原冻结目标特点训练正文')
        self.surrogate = surrogate
        encoder = ChunkedClipEncoder(SimpleNamespace(
            surrogate=surrogate, device=surrogate.device, snapshot=surrogate.snapshot()))
        own = encoder.encode_text(source_public_key)
        own_audit = copy.deepcopy(encoder.last_encoding_audit)
        text = encoder.encode_text(training_queries[0])
        text_audit = copy.deepcopy(encoder.last_encoding_audit)
        self.title_feature = torch.as_tensor(own, device=surrogate.device, dtype=torch.float32).reshape(1, -1).detach()
        self.training_text_feature = torch.as_tensor(text, device=surrogate.device, dtype=torch.float32).reshape(1, -1).detach()
        self.target_feature = F.normalize(target_feature.detach().float(), dim=-1)
        self.query_features = F.normalize(self.target_feature + self.training_text_feature, dim=-1).detach()
        teachers, valid = compensated_targets(self.query_features, self.title_feature)
        if not bool(valid.all()):
            raise ValueError('训练联合查询与自身公开全文键不在正半球，解析补偿教师无效')
        self.compensated_features = teachers.detach()
        self.audit = dict(
            algorithm='public_fullkey_joint_and_analytic_compensated_image_v1',
            source_public_key_text=source_public_key,
            source_public_key_sha256=hashlib.sha256(source_public_key.encode('utf-8')).hexdigest(),
            source_public_key_audit=own_audit, training_query_audit=text_audit,
            training_queries=training_queries, query_image='target_image_unpurified',
            query_image_text_weight=[0.5, 0.5], key_image_text_weight=[0.5, 0.5],
            joint_compensated_objective_weight=[0.5, 0.5],
            compensated_formula='unit(2*dot(query,own)*query-own)',
            query_own_cosine=(self.query_features @ self.title_feature.T).cpu().tolist(),
            compensation_valid=valid.cpu().tolist(),
            private_gallery_used=False, victim_memory_used=False, formal_queries_used=False)

    def scores(self, features):
        key = F.normalize(F.normalize(features.float(), dim=-1) + self.title_feature, dim=-1)
        return key @ self.query_features.T

    def compensated_scores(self, features):
        return F.normalize(features.float(), dim=-1) @ self.compensated_features.T

    def diagnostics(self, features):
        features = F.normalize(features.float(), dim=-1)
        return dict(joint_scores=self.scores(features).detach().cpu().tolist(),
                    compensated_image_cosine=self.compensated_scores(features).detach().cpu().tolist(),
                    image_to_target_cosine=(features @ self.target_feature.T).detach().cpu().tolist(),
                    image_to_training_text_cosine=(features @ self.training_text_feature.T).detach().cpu().tolist(),
                    image_to_query_cosine=(features @ self.query_features.T).detach().cpu().tolist())


def _files(paths):
    result = {}
    for value in paths:
        path = Path(value).resolve()
        digest = sha256_file(path)
        if not digest:
            raise FileNotFoundError('缺少绑定文件：' + str(path))
        result[str(path)] = digest
    return result


def validate_prefix(prefix_root, definition):
    """只读核对本项目同组原完整 PerLAGE_no_pixel 结果及前三阶段；不读取正式查询或记忆库。"""
    validate_definition(definition)
    data = training_data(definition)
    root = Path(prefix_root).resolve()
    root.relative_to(PROJECT.resolve())
    original = validate_original_complete(_read(root / 'ready.json'))
    if not original.get('method_identity_sha256') or sha256_file(original.get('method_identity')) != original['method_identity_sha256']:
        raise ValueError('复用前缀缺少真实方法身份摘要')
    for role in ('source', 'target'):
        if original[role + '_sha256'] != sha256_file(data[role]):
            raise ValueError('复用前缀的公开图对不一致：' + role)
    title_digest = hashlib.sha256(data['source_title'].encode('utf-8')).hexdigest()
    if original['source_title_sha256'] != title_digest:
        raise ValueError('复用前缀的公开源商品名改变')
    initial = _read(original['initialization_identity'])
    if initial['training_queries'] != data['training_queries'] or initial['surface_regions'] != definition['print_regions']:
        raise ValueError('复用前缀的训练正文或印刷保护区改变')
    for role in ('source', 'target'):
        segmentation = _read(root / 'foreground_segmentation' / role / 'record.json')
        if segmentation['identity']['boxes'] != definition[role + '_boxes']:
            raise ValueError('复用前缀的分割提示改变：' + role)
    init_opt = _read(original['initialization_optimizer_identity'])
    if any(init_opt['config'][key] != 0 for key in ('pixel_clean_steps', 'pixel_diffusion_steps', 'pixel_levels')):
        raise ValueError('复用前缀存在像素优化')
    anchor = root / 'color_harmonization/generative_anchor.png'
    initial_image = root / 'color_harmonization/initial.png'
    perlage_final = root / 'adversarial.png'
    if not np.array_equal(read_rgb(anchor), read_rgb(initial_image)):
        raise ValueError('复用前缀存在固定像素残差')
    if sha256_file(perlage_final) != original['candidates']['final']['sha256']:
        raise ValueError('复用 PerLAGE_no_pixel 最终成图摘要不符')
    init_ready = _read(root / 'initialization/ready.json')
    teacher = Path(init_ready['candidates']['natural_final']['image'])
    parent_anchor = Path(init_ready['candidates']['generation_only']['image'])
    for key, record in init_ready['candidates'].items():
        if sha256_file(record['image']) != record['sha256']:
            raise ValueError('复用初始化图像改变：' + key)
    binding = {
        'root': str(root), 'ready_sha256': sha256_file(root / 'ready.json'),
        'perlage_no_pixel_final_sha256': sha256_file(perlage_final),
        'first_three_stages_reused': True, 'fresh_first_three_stages_executed': False,
        'initialization_reused': True, 'subject_locked_to_perlage_no_pixel': True,
        'stages': {name: copy.deepcopy(original['stages'][name]) for name in STAGES[:3]},
        'initialization_evidence': original['initialization_evidence'],
        'files': _files([
            root / 'ready.json', original['initialization_identity'],
            original['initialization_optimizer_identity'], root / 'initialization/ready.json',
            anchor, initial_image, perlage_final, teacher, parent_anchor,
            root / 'color_harmonization/edit_mask.png',
            root / 'foreground_segmentation/source/foreground.png']),
    }
    return root, original, data, teacher, parent_anchor, perlage_final, _canonical(binding)


class _VaeOnly:
    """同一冻结自编码器，只提供解码路线，不加载或调用噪声网络。"""
    def __init__(self, device):
        from diffusers import AutoencoderKL
        self.device = torch.device(device)
        root = Path(model_path('stable-diffusion-v1-5')).resolve()
        self.vae = AutoencoderKL.from_pretrained(
            str(root), subfolder='vae', local_files_only=True,
            use_safetensors=True, variant='fp16', torch_dtype=torch.float32)
        self.vae.eval().requires_grad_(False).to(self.device)
        self.vae.enable_gradient_checkpointing()
        self.identity = {
            'kind': 'same_sd15_vae_only_no_noise_network',
            'files': _files(p for p in sorted((root / 'vae').glob('*'))
                            if p.is_file() and p.suffix in {'.json', '.safetensors', '.bin'}),
            'precision': 'float32', 'posterior': 'mode', 'diffusion_calls': 0,
        }

    def _decode_raw(self, latent, shape):
        image = self.vae.decode(latent / self.vae.config.scaling_factor).sample.float() / 2 + 0.5
        return F.interpolate(image, size=shape[-2:], mode='bicubic', align_corners=False, antialias=True)

    def _decode(self, latent, shape):
        return self._decode_raw(latent, shape).clamp(0, 1)


def _make_sep_gaussian(sigma, device, channels):
    if sigma <= 0:
        return None
    radius = int(np.ceil(3.0 * sigma))
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = g / g.sum()
    kh = g.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    kv = g.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    return kh, kv, radius


def _smooth_sep(x, kh, kv, radius, groups):
    padded = F.pad(x, (radius, radius, radius, radius), mode='reflect')
    out = F.conv2d(padded, kh, groups=groups)
    return F.conv2d(out, kv, groups=groups)


class _BackgroundLatentBranch:
    """商品主体锁定为 PerLAGE_no_pixel 成图、仅在背景潜空间解码残差的分支；零潜变量精确返回零编辑。"""
    def __init__(self, source, eps_map, tau_map, support_mask, purifier, config):
        self.source = source
        self.eps_map = eps_map
        self.tau_map = tau_map
        self.support_mask = support_mask
        self.purifier = purifier
        self.config = config
        self.max_eps = float(max(config.background_levels, config.latent_levels) - 0.01) / 255.0
        self.luma_kernel = _make_sep_gaussian(config.bg_luma_smooth_sigma, source.device, 1)
        self.chroma_kernel = _make_sep_gaussian(config.bg_chroma_smooth_sigma, source.device, 3)
        with torch.no_grad():
            from src.visual.diffusion_purification import working_shape
            work = F.interpolate(source, size=working_shape(source.shape, config.latent_working_size),
                mode='bicubic', align_corners=False, antialias=True).clamp(0, 1)
            self.base = (purifier.vae.encode(work * 2 - 1).latent_dist.mode().float()
                * purifier.vae.config.scaling_factor).detach()
            self.reference = purifier._decode_raw(self.base, source.shape).detach()
        self.delta = torch.zeros_like(self.base, requires_grad=True)

    def residual(self):
        decoded = self.purifier._decode_raw(self.base + self.delta, self.source.shape)
        diff = decoded - self.reference
        if self.config.bg_chroma_keep < 1.0:
            mean_c = diff.mean(dim=1, keepdim=True)
            diff = (1.0 - self.config.bg_chroma_keep) * mean_c + self.config.bg_chroma_keep * diff
        bounded = self.eps_map * torch.tanh(diff / (self.tau_map * self.config.tau_scale))
        if self.luma_kernel is not None or self.chroma_kernel is not None:
            luma = bounded.mean(dim=1, keepdim=True)
            chroma = bounded - luma
            if self.luma_kernel is not None:
                luma = _smooth_sep(luma, *self.luma_kernel, groups=1)
            if self.chroma_kernel is not None:
                chroma = _smooth_sep(chroma, *self.chroma_kernel, groups=3)
            bounded = (luma + chroma).clamp(-self.max_eps, self.max_eps) * self.support_mask
        return bounded


def validate_complete(ready, image=None):
    if ready.get('method') != METHOD or ready.get('definition') != DEFINITION or ready.get('status') != 'completed':
        raise ValueError('独立背景优化试验尚未完整完成')
    if ready.get('pixel_optimization') is not False or ready.get('auxiliary_max_levels') != 0:
        raise ValueError('独立背景优化试验出现像素优化或残差')
    for name in ('victim_memory_used', 'formal_queries_used', 'shadow_pool_used', 'true_rank_used', 'purchase_history_used'):
        if ready.get(name) is not False:
            raise ValueError('独立背景优化试验违反训练边界：' + name)
    if ready.get('first_three_stages_reused') is not True or ready.get('fresh_first_three_stages_executed') is not False:
        raise ValueError('未如实记录冻结前三阶段复用')
    if ready.get('subject_identical_to_perlage_no_pixel') is not True:
        raise ValueError('未验证商品主体与 PerLAGE_no_pixel 完全一致')
    stages = ready.get('stages', {})
    if set(stages) != set(STAGES):
        raise ValueError('缺少四阶段完整证据')
    for stage in stages.values():
        if stage.get('status') != 'completed' or not stage.get('files'):
            raise ValueError('阶段未完成')
        for path, digest in stage['files'].items():
            if not digest or sha256_file(path) != digest:
                raise ValueError('阶段证据改变：' + path)
    last = stages['latent_generation_refinement']
    if type(last.get('steps_completed')) is not int or last['steps_completed'] < 1 or last['steps_completed'] != last.get('steps_requested'):
        raise ValueError('未完成全部细化步数')
    identity_path = ready['method_identity']
    if not ready.get('method_identity_sha256') or sha256_file(identity_path) != ready['method_identity_sha256']:
        raise ValueError('方法身份改变')
    identity = _read(identity_path)
    if identity.get('method') != METHOD or identity.get('definition') != DEFINITION or identity.get('profile') != ready.get('profile'):
        raise ValueError('完成记录与独立方法身份不一致')
    config = identity['config']
    BgCoopLatentConfig(**config).validate()
    if ready.get('latent_config') != config or last['steps_requested'] != config['steps']:
        raise ValueError('完成记录的档位配置或步数改变')
    if ready.get('prefix_binding') != identity['prefix_binding']:
        raise ValueError('完成记录的公共前缀绑定改变')
    if ready.get('training_data_policy') != TRAINING_DATA_POLICY:
        raise ValueError('完成记录的训练输入策略改变')
    public_data = training_data(identity['public_definition'])
    own_key = build_source_public_key(identity['source_public_record'], public_data['source_title'])
    if identity.get('source_public_key_text') != own_key or ready.get('source_public_key_text') != own_key:
        raise ValueError('完成记录的真实公开全文键改变')
    if identity.get('halo_hard_gate') is not False or identity.get('mask_policy') != MASK_POLICY:
        raise ValueError('完成记录的背景优化训练方案改变')
    for role in ('source', 'target'):
        path = str(Path(public_data[role]).resolve())
        expected = identity['inputs'].get(path)
        if not expected or expected != ready.get(role + '_sha256') or sha256_file(path) != expected:
            raise ValueError('完成记录的公开图对绑定改变：' + role)
    title_digest = hashlib.sha256(public_data['source_title'].encode('utf-8')).hexdigest()
    if ready.get('source_title_sha256') != title_digest:
        raise ValueError('完成记录的公开源商品名绑定改变')
    if identity.get('pixel_optimization') is not False or identity.get('auxiliary_levels') != 0 or identity.get('diffusion_calls') != 0:
        raise ValueError('方法身份违反零像素及无净化条件')
    for path, digest in identity['prefix_binding']['files'].items():
        if sha256_file(path) != digest:
            raise ValueError('冻结公共前缀改变')
    candidate = ready['candidates']['final']
    if not candidate.get('sha256') or sha256_file(candidate['image']) != candidate['sha256'] or (image is not None and sha256_file(image) != candidate['sha256']):
        raise ValueError('最终图片摘要改变')
    if not np.array_equal(read_rgb(candidate['image']), read_rgb(ready['generative_anchor'])):
        raise ValueError('最终图片存在像素加性残差')
    prefix_root = Path(identity['prefix_binding']['root'])
    perlage_rgb = read_rgb(prefix_root / 'adversarial.png')
    foreground = read_rgb(prefix_root / 'foreground_segmentation/source/foreground.png')[..., 0] > 0
    final_rgb = read_rgb(candidate['image'])
    if not np.array_equal(final_rgb[foreground], perlage_rgb[foreground]):
        raise ValueError('商品主体前景未与 PerLAGE_no_pixel 保持逐像素完全一致')
    return ready


validate_bg_coop_complete = validate_complete


def generate_bg_coop_no_pixel(source, target, output, *, definition, prefix_root,
                              source_public_record, profile='fg_lock_bg24', device='cuda', pause_path=None):
    """按预注册档位执行商品主体锁定（同 PerLAGE_no_pixel）＋背景 350 步潜空间优化。"""
    if profile not in BG_COOP_PRESETS:
        raise ValueError('未知的预定义背景优化档位')
    config = BgCoopLatentConfig(**BG_COOP_PRESETS[profile])
    config.validate()
    prefix, original_ready, data, teacher, parent_anchor, perlage_final, prefix_binding = validate_prefix(prefix_root, definition)
    for role, given in (('source', source), ('target', target)):
        if sha256_file(given) != sha256_file(data[role]):
            raise ValueError('调用图对和冻结公开配置不一致')
    source_public_record = _canonical(source_public_record)
    source_public_key_text = build_source_public_key(source_public_record, data['source_title'])
    root = Path(output).resolve()
    root.relative_to(PROJECT.resolve())
    root.mkdir(parents=True, exist_ok=True)
    tick = time.perf_counter()
    pause_path = Path(pause_path) if pause_path else root / 'PAUSE_REQUESTED'
    if pause_path.exists():
        raise InterruptedError('独立背景优化试验已暂停')
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    from src.visual.encoders.clip import make_encoder
    base = make_encoder(device=device)
    surrogate = base.surrogate
    device = surrogate.device
    vae = _VaeOnly(device)
    implementation_paths = [Path(__file__), Path(__file__).with_name('__init__.py')]
    implementation_paths += [PROJECT / p for p in (
        'src/visual/_perlage/components/latent_branch.py',
        'src/visual/_perlage/components/core.py', 'src/visual/_perlage/components/colour.py',
        'src/visual/retrieval_objective.py', 'src/visual/robust_objectives.py',
        'src/visual/target_features.py',
        'src/visual/image_utils.py', 'src/visual/encoders/backbone.py',
        'src/visual/encoders/clip.py', 'src/visual/encoders/joint.py')]
    identity = _canonical(dict(
        method=METHOD, definition=DEFINITION, profile=profile, config=asdict(config),
        public_definition=definition, source_public_record=source_public_record,
        source_public_key_text=source_public_key_text, prefix_binding=prefix_binding,
        inputs=_files([source, target]), encoder=surrogate.snapshot(), vae=vae.identity,
        implementation=_files(implementation_paths),
        training_data_policy=TRAINING_DATA_POLICY, training_queries=data['training_queries'],
        objective='equal_absolute_fullkey_joint_and_analytic_compensated_image_cosine_background_only_v2',
        image_text_training_weight=[0.5, 0.5], feature_retention=False,
        ordinary_teacher_score_floor=False, noise_retention_gate=False,
        diffusion_calls=0, pixel_optimization=False, auxiliary_levels=0,
        victim_memory_used=False, formal_queries_used=False, shadow_pool_used=False,
        true_rank_used=False, selection_rule='best_public_training_objective_with_fixed_quality_gates',
        halo_hard_gate=False, mask_policy=MASK_POLICY))
    identity_path = root / 'method_identity.json'
    if identity_path.exists():
        if _read(identity_path) != identity:
            raise ValueError('独立试验身份改变，须使用新目录')
    elif any(root.iterdir()):
        raise ValueError('拒绝覆盖没有独立身份的非空目录')
    else:
        write_json(identity_path, identity)
    if (root / 'ready.json').exists():
        ready = validate_complete(_read(root / 'ready.json'))
        del base, vae
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return Path(ready['candidates']['final']['image']), ready

    original = read_tensor(source, device)
    # 锚点直接取同组已完成 PerLAGE_no_pixel 的最终图像，使前景主体像素与 PerLAGE_no_pixel 100% 一致
    anchor = read_tensor(perlage_final, device)
    oldmask = np.asarray(Image.open(relocate_io_path(prefix / 'color_harmonization/edit_mask.png')).convert('L')).copy()
    foreground = read_rgb(prefix / 'foreground_segmentation/source/foreground.png')[..., 0] > 0
    roi = print_region_mask(read_rgb(source).shape, definition['print_regions'])
    dist_out = distance_transform_edt(~foreground)
    if config.edge_feather_mode == 'cosine':
        t_ramp = np.clip(dist_out / config.edge_feather_pixels, 0.0, 1.0)
        outward_bg = 0.5 * (1.0 - np.cos(np.pi * t_ramp)) * (~foreground)
    elif config.edge_feather_mode == 'exponential':
        outward_bg = (1.0 - np.exp(-(dist_out ** 2) / (2.0 * (config.edge_feather_pixels ** 2)))) * (~foreground)
    else:
        outward_bg = np.clip(dist_out / config.edge_feather_pixels, 0.0, 1.0) * (~foreground)
    inward_fg = np.clip(distance_transform_edt(foreground) / config.edge_feather_pixels, 0.0, 1.0) * foreground
    level_array = (config.background_levels * outward_bg + config.latent_levels * inward_fg) * (~roi)
    max_level = float(max(config.background_levels, config.latent_levels, 1))
    macro_array = level_array / max_level
    eps_map = torch.as_tensor(level_array / 255.0, device=device, dtype=torch.float32)[None, None]
    tau_floor = max(config.background_levels, 1) / 255.0
    tau_map = eps_map.clamp_min(tau_floor)
    fg_bool = torch.as_tensor(foreground, device=device, dtype=torch.bool)
    bg_bool = ~fg_bool
    bg_mask_t = torch.as_tensor(((~foreground) & (~roi)).astype(np.float32), device=device)[None, None]
    Image.fromarray(np.rint(macro_array * 255).astype(np.uint8)).save(root / 'latent_edit_mask.png')

    context = prepare_print_context(read_rgb(source), read_rgb(parent_anchor), oldmask, foreground, definition['print_regions'])
    parent_halo = halo_error(context, read_rgb(teacher))
    halo_ceiling = max(parent_halo * config.halo_parent_multiplier, config.halo_floor)
    hy, hx = np.where(context['halo'])
    nearest_y, nearest_x = context['nearest']
    halo_y = torch.as_tensor(hy, device=device, dtype=torch.long)
    halo_x = torch.as_tensor(hx, device=device, dtype=torch.long)
    near_y = torch.as_tensor(nearest_y[hy, hx], device=device, dtype=torch.long)
    near_x = torch.as_tensor(nearest_x[hy, hx], device=device, dtype=torch.long)

    def differentiable_halo(image):
        if not len(hy):
            return image.sum() * 0
        difference = (image - original)[0] * 255
        jumps = difference[:, halo_y, halo_x] - difference[:, near_y, near_x]
        return torch.linalg.vector_norm(jumps, dim=0).mean()

    with torch.no_grad():
        target_feature = surrogate.encode_image(read_tensor(target, device)).global_embedding.detach()
    joint = PublicFullKeyObjective(surrogate, target_feature, source_public_key_text, data['training_queries'])
    write_json(root / 'public_title_encoding_audit.json', joint.audit)
    branch = _BackgroundLatentBranch(anchor, eps_map, tau_map, bg_mask_t, vae, config)
    optimizer = torch.optim.Adam([branch.delta], lr=config.latent_learning_rate)

    def compose_with_raw():
        raw = anchor + branch.residual()
        return quantize_ste(raw.clamp(0, 1)), raw

    def compose():
        return quantize_ste((anchor + branch.residual()).clamp(0, 1))

    def value_and_penalty(image, raw=None):
        features = surrogate.encode_image(image).global_embedding
        scores = joint.scores(features)
        utility = 0.5 * scores.mean() + 0.5 * softmin(scores.flatten(), config.temperature)
        comp_scores = joint.compensated_scores(features)
        comp_utility = 0.5 * comp_scores.mean() + 0.5 * softmin(comp_scores.flatten(), config.temperature)
        utility = config.joint_weight * utility + config.compensated_weight * comp_utility

        bg_denom = max(config.background_levels, 1) / 255.0
        bg_unit = ((image - anchor) * bg_mask_t) / bg_denom
        bg_tv = (bg_unit[..., 1:, :] - bg_unit[..., :-1, :]).abs().mean() + (bg_unit[..., :, 1:] - bg_unit[..., :, :-1]).abs().mean()
        bg_chroma = bg_unit - bg_unit.mean(dim=1, keepdim=True)
        bg_chroma_tv = (bg_chroma[..., 1:, :] - bg_chroma[..., :-1, :]).abs().mean() + (bg_chroma[..., :, 1:] - bg_chroma[..., :, :-1]).abs().mean()
        bg_l2 = (bg_unit.square().sum() / bg_mask_t.sum().clamp_min(1.0) / 3.0)
        overflow = (F.relu(raw - 1.0).square().mean() + F.relu(-raw).square().mean()) if raw is not None else (image.sum() * 0.0)

        penalty = (config.bg_l2_weight * bg_l2
                   + config.bg_tv_weight * bg_tv
                   + config.bg_chroma_tv_weight * bg_chroma_tv
                   + config.overflow_weight * overflow
                   + config.halo_soft_weight * differentiable_halo(image))
        return utility, penalty, scores

    @torch.no_grad()
    def quality(image):
        ss = float(source_ssim(image, original))
        diff_orig = (image - original).abs() * 255.0
        diff_anchor = (image - anchor).abs() * 255.0
        mae = float(diff_orig.mean())
        fg_source_mae = float(diff_orig[:, :, fg_bool].mean())
        bg_source_mae = float(diff_orig[:, :, bg_bool].mean())
        fg_anchor_mae = float(diff_anchor[:, :, fg_bool].mean())
        bg_anchor_mae = float(diff_anchor[:, :, bg_bool].mean())
        arr = np.rint(image[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        halo = halo_error(context, arr)
        return dict(source_ssim=ss, source_mae_levels=mae,
                    fg_source_mae_levels=fg_source_mae, bg_source_mae_levels=bg_source_mae,
                    fg_anchor_mae_levels=fg_anchor_mae, bg_anchor_mae_levels=bg_anchor_mae,
                    halo_error=halo,
                    quality_ok=ss >= config.source_ssim_floor and mae <= config.source_mae_ceiling,
                    natural_halo_gate=halo <= halo_ceiling, halo_hard_gate=False)

    @torch.no_grad()
    def development(image, step):
        row = quality(image)
        utility, penalty, scores = value_and_penalty(image)
        row.update(step=step, joint_scores=scores.cpu().tolist(), training_utility=float(utility),
                   regularization=float(penalty), selection_utility=float(utility - penalty),
                   halo_ceiling=halo_ceiling, parent_halo_error=parent_halo)
        features = surrogate.encode_image(image).global_embedding
        row.update(joint.diagnostics(features))
        row['selection'] = [int(row['quality_ok']), row['selection_utility']]
        return row

    history, best_row, best = [], None, None
    step_done, previous, accepted, rejected = 0, 0.0, 0, 0
    checkpoint_path = root / 'checkpoint.pt'
    if checkpoint_path.exists():
        state = torch.load(relocate_io_path(checkpoint_path), map_location=device, weights_only=False)
        if state['identity'] != identity:
            raise ValueError('独立试验断点身份改变')
        with torch.no_grad():
            branch.delta.copy_(state['latent'])
        optimizer.load_state_dict(state['optimizer'])
        history, best_row, best = state['history'], state['best_row'], state['best']
        step_done, previous = state['step'], state['seconds']
        accepted, rejected = state['accepted'], state['rejected']

    def save(status):
        elapsed = previous + time.perf_counter() - tick
        pending = checkpoint_path.with_suffix('.tmp')
        torch.save(dict(identity=identity, latent=branch.delta.detach(), optimizer=optimizer.state_dict(),
                        history=history, best_row=best_row, best=best, step=step_done,
                        seconds=elapsed, accepted=accepted, rejected=rejected), relocate_io_path(pending))
        pending.replace(checkpoint_path)
        write_json(root / 'progress.json', dict(status=status, profile=profile, step=step_done,
                   requested_steps=config.steps, best=best_row, accepted_updates=accepted,
                   rejected_updates=rejected, seconds=elapsed))

    if best_row is None:
        best = branch.delta.detach().clone()
        best_row = development(compose(), 0)
        if best_row['selection'][0] != 1:
            write_json(root / 'initial_quality_rejection.json', best_row)
            raise ValueError('冻结起点未通过本轮固定质量守门')
        history.append(best_row)
        save('running')
    for step in range(step_done + 1, config.steps + 1):
        if pause_path.exists():
            save('paused')
            raise InterruptedError('独立背景优化试验已保存细化断点')
        image = compose().detach().requires_grad_(True)
        normals, slacks = [], []
        structure = source_ssim(image, original)
        normal, = torch.autograd.grad(structure, image)
        normals.append(normal.detach())
        slacks.append(float(structure.detach()) - config.source_ssim_floor)
        average = (image - original).abs().mean()
        normal, = torch.autograd.grad(-average, image)
        normals.append(normal.detach())
        slacks.append(config.source_mae_ceiling / 255 - float(average.detach()))
        old = branch.delta.detach().clone()
        old_optimizer = copy.deepcopy(optimizer.state_dict())
        optimizer.zero_grad(set_to_none=True)
        decoded, raw_decoded = compose_with_raw()
        latent_normals = []
        for normal in normals:
            derivative, = torch.autograd.grad(decoded, branch.delta, grad_outputs=normal, retain_graph=True)
            latent_normals.append(derivative.detach())
        utility, penalty, _ = value_and_penalty(decoded, raw=raw_decoded)
        loss = -utility + penalty
        loss.backward()
        if branch.delta.grad is None or not torch.isfinite(branch.delta.grad).all():
            raise FloatingPointError('独立试验潜变量梯度非有限')
        optimizer.param_groups[0]['lr'] = config.latent_learning_rate * (1 - 0.75 * step / config.steps)
        optimizer.step()
        with torch.no_grad():
            branch.delta.clamp_(-config.latent_limit, config.latent_limit)
            smooth = F.avg_pool2d(F.pad(branch.delta, (1, 1, 1, 1), mode='reflect'), 3, stride=1)
            if config.latent_smoothing_mix:
                branch.delta.lerp_(smooth, config.latent_smoothing_mix)
            projected = tangent_projection(branch.delta - old, latent_normals, slacks)
            proposed = (old + projected).clamp(-config.latent_limit, config.latent_limit)
            fraction = 0.0
            for backtrack in range(config.guard_backtracks + 1):
                scale = 0.5 ** backtrack
                branch.delta.copy_(old + scale * (proposed - old))
                q = quality(compose())
                if q['quality_ok']:
                    fraction = scale
                    break
            if fraction:
                accepted += 1
            else:
                branch.delta.copy_(old)
                optimizer.load_state_dict(old_optimizer)
                rejected += 1
        step_done = step
        if step % config.check_every == 0:
            row = development(compose(), step)
            history.append(row)
            save_tensor_image(root / f'checkpoint_{step:04d}.png', compose())
            if tuple(row['selection']) > tuple(best_row['selection']):
                best, best_row = branch.delta.detach().clone(), row
            save('running')
            print(json.dumps(dict(profile=profile, step=step, best_step=best_row['step'],
                  utility=row['training_utility'], source_ssim=row['source_ssim'],
                  source_mae_levels=row['source_mae_levels'],
                  fg_source_mae_levels=row['fg_source_mae_levels'],
                  bg_source_mae_levels=row['bg_source_mae_levels'],
                  accepted=accepted, rejected=rejected), ensure_ascii=False), flush=True)
    with torch.no_grad():
        branch.delta.copy_(best)
        final_tensor = compose()
        final_check = development(final_tensor, best_row['step'])
    if final_check['selection'][0] != 1:
        raise AssertionError('最终图像未通过固定质量守门')
    final = root / 'adversarial.png'
    generative_anchor = root / 'generative_anchor.png'
    save_tensor_image(final, final_tensor)
    save_tensor_image(generative_anchor, final_tensor)
    final_rgb, anchor_rgb = read_rgb(final), read_rgb(perlage_final)
    change = np.abs(final_rgb.astype(np.int16) - anchor_rgb.astype(np.int16))
    if (np.any(change[roi])
            or int(change[foreground].max()) > config.latent_levels
            or int(change[~foreground].max()) > config.background_levels
            or not np.array_equal(final_rgb[foreground], anchor_rgb[foreground])):
        raise AssertionError('商品主体锁定（同 PerLAGE_no_pixel）、背景预算或印刷区域保护失败')
    metadata = dict(status='completed', config=asdict(config), history=history,
        steps_completed=step_done, best=best_row, final_check=final_check,
        accepted_updates=accepted, rejected_updates=rejected,
        latent_max_levels=int(change.max()), latent_mae_levels=float(change.mean()),
        fg_latent_max_levels=int(change[foreground].max()), fg_latent_mae_levels=float(change[foreground].mean()),
        bg_latent_max_levels=int(change[~foreground].max()), bg_latent_mae_levels=float(change[~foreground].mean()),
        subject_identical_to_perlage_no_pixel=True,
        auxiliary_max_levels=0, auxiliary_mae_levels=0.0, pixel_optimization=False,
        no_pixel_optimization=True, diffusion_calls=0, printed_region_fixed_to_harmonized_anchor=True,
        halo_hard_gate=False, halo_soft_weight=config.halo_soft_weight,
        mask_nonzero_pixels=int((macro_array > 0).sum()),
        mask_mean=float(macro_array.mean()), full_foreground_mask=False,
        foreground_limited=False, background_only_optimization=True,
        private_gallery_used=False, true_rank_used=False,
        elapsed_seconds=previous + time.perf_counter() - tick,
        output_sha256=sha256_file(final))
    write_json(root / 'metadata.json', metadata)
    stages = copy.deepcopy(prefix_binding['stages'])
    for record in stages.values():
        record['execution_mode'] = 'reused_frozen_prefix'
    stages[STAGES[-1]] = stage_record([identity_path, root / 'metadata.json', final,
        generative_anchor, root / 'latent_edit_mask.png', root / 'public_title_encoding_audit.json'],
        steps_completed=step_done, steps_requested=config.steps, execution_mode='fresh_full_refinement')
    ready = dict(method=METHOD, definition=DEFINITION, profile=profile, status='completed',
        color_scheme='statistics', training_data_policy=TRAINING_DATA_POLICY,
        first_three_stages_reused=True, fresh_first_three_stages_executed=False,
        initialization_reused=True, previous_final_refinement_reused=False,
        subject_identical_to_perlage_no_pixel=True,
        method_identity=str(identity_path), method_identity_sha256=sha256_file(identity_path),
        source_sha256=sha256_file(source), target_sha256=sha256_file(target),
        source_title_sha256=original_ready['source_title_sha256'],
        stages=stages, latent_config=asdict(config), source_public_key_text=source_public_key_text,
        candidates={'final': {'image': str(final), 'sha256': sha256_file(final)}},
        generative_anchor=str(generative_anchor), prefix_binding=prefix_binding,
        pixel_optimization=False, auxiliary_max_levels=0, optimization_mode='latent_only',
        victim_memory_used=False, formal_queries_used=False, shadow_pool_used=False,
        true_rank_used=False, purchase_history_used=False, diffusion_calls=0,
        observed_seconds=metadata['elapsed_seconds'])
    validate_complete(ready, final)
    write_json(root / 'ready.json', ready)
    save('completed')
    del base, vae, surrogate, branch, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return final, ready
