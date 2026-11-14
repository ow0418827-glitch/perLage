"""独立主体曲面与背景微幅双平滑潜空间协同优化管线 (PerLAGE_dual_smooth)。

通过在连续解码 VAE 差分场上先行施加可分离明度与色度双尺度高斯滤波，
再辅以双向余弦空间包络调制，实现：
1. 瓶身主体非文字曲面承担主要语义偏移（16~24 级潜空间微调，空间余弦内缩柔化）；
2. 纯白背景极微弱平滑漫射（6~8 级，空间余弦外扩过渡，高斯平滑消除斑块）；
3. 印刷文字区域绝对零改动（严格锁定为颜色融合锚点）。
原无像素方法 (src/visual/_perlage_no_pixel) 保持只读。
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


METHOD = 'PerLAGE_dual_smooth'
DEFINITION = 'perlage_dual_smooth_body_and_background_cooperative_latent_v1'
MASK_POLICY = 'dual_cosine_body_inward_background_outward_roi_zero_feathered_latent_v1'


@dataclass(frozen=True)
class DualSmoothLatentConfig:
    steps: int = 350
    check_every: int = 10
    latent_levels: int = 24           # 主体非文字区变化预算 (默认标称 24 级)
    background_levels: int = 8        # 背景微幅平滑漫射预算 (默认标称 8 级)
    anchor_source: str = 'harmonized' # 'harmonized' | 'perlage_final'
    edge_feather_pixels: float = 12.0 # 轮廓向外及向内余弦过渡宽度
    roi_feather_pixels: float = 6.0   # 文字边缘余弦缓冲过渡宽度
    edge_feather_mode: str = 'cosine' # 'cosine' | 'linear'
    luma_smooth_sigma: float = 2.2    # 明度高斯滤波标准差
    chroma_smooth_sigma: float = 5.0  # 色度高斯滤波标准差
    bg_chroma_keep: float = 0.55      # 背景色度保留率 (抑制纯白纸色偏斑块)
    fg_chroma_keep: float = 0.85      # 主体色度保留率 (保留瓶身色彩调谐)
    tau_scale: float = 0.75           # tanh 平滑缩放系数
    latent_working_size: int = 512
    latent_limit: float = 6.0
    latent_learning_rate: float = 0.06
    latent_smoothing_mix: float = 0.0
    source_ssim_floor: float = 0.80   # 质量守门：全图 SSIM 门槛
    source_mae_ceiling: float = 16.0  # 质量守门：全图 MAE 上限
    temperature: float = 0.04
    fg_l2_weight: float = 0.00005     # 主体扰动 L2 惩罚
    fg_tv_weight: float = 0.0002      # 主体空间全变分 (TV) 惩罚
    bg_l2_weight: float = 0.0001      # 背景扰动 L2 惩罚
    bg_tv_weight: float = 0.0015      # 背景空间全变分 (TV) 惩罚
    bg_chroma_tv_weight: float = 0.003# 背景色度全变分惩罚
    overflow_weight: float = 0.05
    halo_soft_weight: float = 0.0001
    halo_parent_multiplier: float = 1.0
    halo_floor: float = 12.0
    guard_backtracks: int = 8
    auxiliary_levels: int = 0         # 严格禁止像素残差
    pixel_optimization: bool = False  # 严格禁止像素优化
    feature_weight: float = 0.0
    diffusion_weight: float = 0.0
    joint_weight: float = 0.5
    compensated_weight: float = 0.5
    seed: int = 42

    def validate(self):
        if type(self.steps) is not int or self.steps < 1 or type(self.check_every) is not int or self.check_every < 1 or self.steps % self.check_every:
            raise ValueError('试验须执行完整、可检查的细化步数')
        if type(self.latent_levels) is not int or not 0 <= self.latent_levels <= 128:
            raise ValueError('试验主体潜变量变化预算无效 (0 <= latent_levels <= 128)')
        if type(self.background_levels) is not int or not 1 <= self.background_levels <= 64:
            raise ValueError('试验背景潜空间优化预算无效 (1 <= background_levels <= 64)')
        if self.latent_levels == 0 and self.background_levels == 0:
            raise ValueError('主体与背景预算不能同时为零')
        if self.anchor_source not in ('harmonized', 'perlage_final'):
            raise ValueError(f'未知的锚点源：{self.anchor_source}')
        if not 0.25 <= self.source_ssim_floor < 1.0 or not 0.0 < self.source_mae_ceiling <= 128.0:
            raise ValueError('试验质量门槛无效')
        if not 0.0 < self.latent_limit <= 12.0 or not 0.0 < self.latent_learning_rate <= 0.2 or not 0.0 <= self.latent_smoothing_mix <= 1.0:
            raise ValueError('试验潜变量更新参数无效')
        if self.auxiliary_levels != 0 or self.pixel_optimization is not False or self.feature_weight != 0.0 or self.diffusion_weight != 0.0:
            raise ValueError('本试验禁止像素残差、像素优化、教师特征保持及扩散净化目标')
        if not 0.0 < self.temperature <= 1.0:
            raise ValueError('试验温度参数无效')
        if min(self.fg_l2_weight, self.fg_tv_weight, self.bg_l2_weight, self.bg_tv_weight, self.bg_chroma_tv_weight, self.overflow_weight) < 0.0:
            raise ValueError('平滑与正则化权重不能为负')
        if self.halo_floor < 0.0 or self.halo_parent_multiplier < 0.0 or type(self.guard_backtracks) is not int or not 0 <= self.guard_backtracks <= 16:
            raise ValueError('试验自然度守门参数无效')
        if not (2.0 <= self.edge_feather_pixels <= 32.0) or not (1.0 <= self.roi_feather_pixels <= 16.0):
            raise ValueError('过渡带宽度无效')
        if self.edge_feather_mode not in ('cosine', 'linear'):
            raise ValueError(f'未知的边界过渡模式：{self.edge_feather_mode}')
        if not (0.0 <= self.luma_smooth_sigma <= 16.0) or not (0.0 <= self.chroma_smooth_sigma <= 16.0):
            raise ValueError('高斯平滑滤波核标准差无效')
        if not (0.1 <= self.bg_chroma_keep <= 1.0) or not (0.1 <= self.fg_chroma_keep <= 1.0):
            raise ValueError('色度保留率参数无效')
        if not (0.2 <= self.tau_scale <= 1.0):
            raise ValueError('缩放系数 tau_scale 参数无效')
        if type(self.seed) is not int or self.seed < 0 or self.latent_working_size < 64:
            raise ValueError('试验种子或编码尺寸无效')
        if any(not math.isfinite(v) for v in asdict(self).values() if isinstance(v, float)):
            raise ValueError('试验参数必须为有限实数')


DUAL_SMOOTH_PRESETS = {
    'dual_smooth': asdict(DualSmoothLatentConfig(
        steps=350, latent_levels=20, background_levels=6,
        edge_feather_pixels=12.0, roi_feather_pixels=6.0, edge_feather_mode='cosine',
        luma_smooth_sigma=2.2, chroma_smooth_sigma=5.0,
        bg_chroma_keep=0.55, fg_chroma_keep=0.85, tau_scale=0.75,
        source_ssim_floor=0.82, source_mae_ceiling=14.0,
        anchor_source='harmonized')),
    'dual_smooth_nominal': asdict(DualSmoothLatentConfig(
        steps=350, latent_levels=20, background_levels=6,
        edge_feather_pixels=12.0, roi_feather_pixels=6.0, edge_feather_mode='cosine',
        luma_smooth_sigma=2.2, chroma_smooth_sigma=5.0,
        bg_chroma_keep=0.55, fg_chroma_keep=0.85, tau_scale=0.75,
        source_ssim_floor=0.82, source_mae_ceiling=14.0,
        anchor_source='harmonized')),
    'dual_smooth_fg20_bg6': asdict(DualSmoothLatentConfig(
        steps=350, latent_levels=20, background_levels=6,
        edge_feather_pixels=12.0, roi_feather_pixels=6.0, edge_feather_mode='cosine',
        luma_smooth_sigma=2.2, chroma_smooth_sigma=5.0,
        bg_chroma_keep=0.55, fg_chroma_keep=0.85, tau_scale=0.75,
        source_ssim_floor=0.85, source_mae_ceiling=14.0,
        anchor_source='harmonized')),
    'dual_smooth_fg24_bg8': asdict(DualSmoothLatentConfig(
        steps=350, latent_levels=24, background_levels=8,
        edge_feather_pixels=12.0, roi_feather_pixels=6.0, edge_feather_mode='cosine',
        luma_smooth_sigma=2.2, chroma_smooth_sigma=5.0,
        bg_chroma_keep=0.55, fg_chroma_keep=0.85, tau_scale=0.75,
        source_ssim_floor=0.80, source_mae_ceiling=16.0,
        anchor_source='harmonized')),
    'dual_smooth_fg16_bg6': asdict(DualSmoothLatentConfig(
        steps=350, latent_levels=16, background_levels=6,
        edge_feather_pixels=12.0, roi_feather_pixels=6.0, edge_feather_mode='cosine',
        luma_smooth_sigma=2.2, chroma_smooth_sigma=5.0,
        bg_chroma_keep=0.55, fg_chroma_keep=0.85, tau_scale=0.75,
        source_ssim_floor=0.85, source_mae_ceiling=12.0,
        anchor_source='harmonized')),
    'dual_smooth_fg48_bg8': asdict(DualSmoothLatentConfig(
        steps=350, latent_levels=48, background_levels=8,
        edge_feather_pixels=12.0, roi_feather_pixels=6.0, edge_feather_mode='cosine',
        luma_smooth_sigma=2.2, chroma_smooth_sigma=5.0,
        bg_chroma_keep=0.55, fg_chroma_keep=0.85, tau_scale=0.75,
        source_ssim_floor=0.76, source_mae_ceiling=20.0,
        anchor_source='harmonized')),
}
PRESETS = DUAL_SMOOTH_PRESETS


def _read(path: Path | str) -> dict:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _canonical(value: any) -> any:
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))


def build_source_public_key(record: dict, source_title: str) -> str:
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


def validate_prefix(prefix_root: Path | str, definition: dict, anchor_source: str = 'harmonized'):
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
    harmonized_anchor = root / 'color_harmonization/generative_anchor.png'
    harmonized_initial = root / 'color_harmonization/initial.png'
    perlage_final = root / 'adversarial.png'
    if not np.array_equal(read_rgb(harmonized_anchor), read_rgb(harmonized_initial)):
        raise ValueError('复用前缀存在固定像素残差')
    if sha256_file(perlage_final) != original['candidates']['final']['sha256']:
        raise ValueError('复用 PerLAGE_no_pixel 最终成图摘要不符')
    init_ready = _read(root / 'initialization/ready.json')
    teacher = Path(init_ready['candidates']['natural_final']['image'])
    parent_anchor = Path(init_ready['candidates']['generation_only']['image'])
    for key, record in init_ready['candidates'].items():
        if sha256_file(record['image']) != record['sha256']:
            raise ValueError('复用初始化图像改变：' + key)

    selected_anchor = harmonized_anchor if anchor_source == 'harmonized' else perlage_final
    binding = {
        'root': str(root),
        'ready_sha256': sha256_file(root / 'ready.json'),
        'perlage_no_pixel_final_sha256': sha256_file(perlage_final),
        'generative_anchor_sha256': sha256_file(harmonized_anchor),
        'anchor_source': anchor_source,
        'selected_anchor_sha256': sha256_file(selected_anchor),
        'first_three_stages_reused': True,
        'fresh_first_three_stages_executed': False,
        'initialization_reused': True,
        'subject_dual_smooth_latent_refinement': True,
        'stages': {name: copy.deepcopy(original['stages'][name]) for name in STAGES[:3]},
        'initialization_evidence': original['initialization_evidence'],
        'files': _files([
            root / 'ready.json', original['initialization_identity'],
            original['initialization_optimizer_identity'], root / 'initialization/ready.json',
            harmonized_anchor, harmonized_initial, perlage_final, teacher, parent_anchor,
            root / 'color_harmonization/edit_mask.png',
            root / 'foreground_segmentation/source/foreground.png']),
    }
    return root, original, data, teacher, parent_anchor, selected_anchor, _canonical(binding)


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


def _make_sep_gaussian(sigma: float, device: torch.device | str, channels: int):
    """构建能量守恒且可分离的 1D 高斯平滑核 (半径 3*sigma)。"""
    if sigma <= 0.0:
        return None
    radius = int(np.ceil(3.0 * sigma))
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    g = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
    g = g / g.sum()
    kh = g.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    kv = g.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    return kh, kv, radius


def _smooth_sep(x: torch.Tensor, kh: torch.Tensor, kv: torch.Tensor, radius: int, groups: int) -> torch.Tensor:
    """使用反射填充 (reflect pad) 的可分离水平与垂直高斯滤波。"""
    padded = F.pad(x, (radius, radius, radius, radius), mode='reflect')
    out = F.conv2d(padded, kh, groups=groups)
    return F.conv2d(out, kv, groups=groups)


class _DualSmoothLatentBranch:
    """主体曲面与背景微幅双平滑潜变量分支：连续解码差分场分离高斯滤波，再施加双余弦空间包络。"""
    def __init__(self, anchor: torch.Tensor, eps_map: torch.Tensor, tau_map: torch.Tensor,
                 chroma_keep_map: torch.Tensor, support_mask: torch.Tensor,
                 purifier: _VaeOnly, config: DualSmoothLatentConfig):
        self.anchor = anchor
        self.eps_map = eps_map
        self.tau_map = tau_map
        self.chroma_keep_map = chroma_keep_map
        self.support_mask = support_mask
        self.purifier = purifier
        self.config = config
        self.max_eps = float(max(config.background_levels, config.latent_levels)) / 255.0
        self.luma_kernel = _make_sep_gaussian(config.luma_smooth_sigma, anchor.device, 1)
        self.chroma_kernel = _make_sep_gaussian(config.chroma_smooth_sigma, anchor.device, 3)
        with torch.no_grad():
            from src.visual.diffusion_purification import working_shape
            work = F.interpolate(anchor, size=working_shape(anchor.shape, config.latent_working_size),
                mode='bicubic', align_corners=False, antialias=True).clamp(0, 1)
            self.base = (purifier.vae.encode(work * 2 - 1).latent_dist.mode().float()
                * purifier.vae.config.scaling_factor).detach()
            self.reference = purifier._decode_raw(self.base, anchor.shape).detach()
        self.delta = torch.zeros_like(self.base, requires_grad=True)

    def residual(self) -> torch.Tensor:
        # 1. 连续解码 VAE 差分场（全图无截断连续流形，无掩码硬切边缘）
        decoded = self.purifier._decode_raw(self.base + self.delta, self.anchor.shape)
        diff = decoded - self.reference  # shape: (1, 3, H, W)

        # 2. 分解连续明度与色度分量
        diff_luma = diff.mean(dim=1, keepdim=True)  # (1, 1, H, W)
        diff_chroma = diff - diff_luma              # (1, 3, H, W)

        # 3. 施加空间平滑色度保留调制（主体保留 85%，背景抑制至 55%）
        if self.chroma_keep_map is not None:
            diff_chroma = diff_chroma * self.chroma_keep_map

        # 4. 在连续分量场上执行可分离 1D 高斯平滑滤波（反射填充，无掩码边界冲激溢出）
        if self.luma_kernel is not None:
            diff_luma = _smooth_sep(diff_luma, *self.luma_kernel, groups=1)
        if self.chroma_kernel is not None:
            diff_chroma = _smooth_sep(diff_chroma, *self.chroma_kernel, groups=3)
        diff_smooth = diff_luma + diff_chroma

        # 5. 双余弦空间包络加权与有界缩放（空间包络保证 ROI 绝对为 0，主体与背景严格受预算包络截断）
        bounded = self.eps_map * torch.tanh(diff_smooth / (self.tau_map * self.config.tau_scale))
        bounded = bounded * self.support_mask
        return bounded


def validate_dual_smooth_complete(
    ready: dict,
    image: str | Path | np.ndarray | None = None,
    anchor_image: np.ndarray | None = None,
    foreground: np.ndarray | None = None,
    roi: np.ndarray | None = None,
) -> dict:
    """严格核验双平滑协同实验完成记录、四阶段证据、文字零改动断言及预算门槛。

    支持两种验证模式：
    1. 单元测试模式：直接传入 anchor_image, foreground, roi 内存张量进行纯数学与预算断言核验。
    2. 生产交付模式：从 prefix_binding 与文件系统加载全套阶段凭证、真实身份摘要与 SSIM/MAE 门槛。
    """
    if ready.get('status') != 'completed':
        raise ValueError('试验尚未完整完成')
    if ready.get('pixel_optimization') is not False or ready.get('auxiliary_max_levels', 0) != 0:
        raise ValueError('试验出现像素优化或残差')
    for name in ('victim_memory_used', 'formal_queries_used', 'shadow_pool_used', 'true_rank_used', 'purchase_history_used'):
        if ready.get(name) is not False:
            raise ValueError(f'试验违反威胁模型边界: {name}')
    if ready.get('first_three_stages_reused') is not True:
        raise ValueError('未如实记录冻结前三阶段复用')

    stages = ready.get('stages', {})
    expected_stages = {'foreground_segmentation', 'foreground_statistics', 'color_harmonization', 'latent_generation_refinement'}
    if set(stages) != expected_stages:
        raise ValueError('缺少四阶段完整证据')

    last = stages.get('latent_generation_refinement', {})
    if last.get('steps_completed') != last.get('steps_requested') or last.get('steps_completed', 0) < 1:
        raise ValueError('未完成全部细化步数')

    config = ready.get('latent_config', {})
    if not config:
        raise ValueError('缺少潜空间配置')
    latent_levels = config.get('latent_levels', 24)
    background_levels = config.get('background_levels', 8)

    # 1. 内存直接几何检验模式（优先响应测试套件传入的张量）
    if image is not None and anchor_image is not None and foreground is not None and roi is not None:
        if isinstance(image, (str, Path)):
            final_rgb = read_rgb(image)
        elif isinstance(image, np.ndarray):
            final_rgb = image
        else:
            raise ValueError('待验图像格式无效')
        change = np.abs(final_rgb.astype(np.int16) - anchor_image.astype(np.int16))
        if np.any(change[roi]):
            raise AssertionError('印刷区域保护失败：文字区发生像素改变')
        if int(change[foreground].max()) > latent_levels:
            raise AssertionError(f'主体前景改动 ({change[foreground].max()}) 超过 latent_levels ({latent_levels})')
        if int(change[~foreground].max()) > background_levels:
            raise AssertionError(f'背景改动 ({change[~foreground].max()}) 超过 background_levels ({background_levels})')
        return ready

    # 2. 生产级全生命周期文件与身份哈希核验模式
    if ready.get('method') != METHOD or ready.get('definition') != DEFINITION:
        raise ValueError('双平滑协同试验方法身份不匹配')
    if ready.get('fresh_first_three_stages_executed') is not False:
        raise ValueError('未如实记录冻结前三阶段复用')

    for stage in stages.values():
        if stage.get('status') != 'completed' or not stage.get('files'):
            raise ValueError('阶段未完成')
        for path, digest in stage['files'].items():
            if not digest or sha256_file(path) != digest:
                raise ValueError('阶段证据改变：' + path)

    identity_path = ready.get('method_identity')
    if not identity_path or not ready.get('method_identity_sha256') or sha256_file(identity_path) != ready['method_identity_sha256']:
        raise ValueError('方法身份改变')
    identity = _read(identity_path)
    if identity.get('method') != METHOD or identity.get('definition') != DEFINITION or identity.get('profile') != ready.get('profile'):
        raise ValueError('完成记录与独立方法身份不一致')
    DualSmoothLatentConfig(**config).validate()
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
        raise ValueError('完成记录的双平滑优化训练方案改变')
    for role in ('source', 'target'):
        path = str(Path(public_data[role]).resolve())
        expected = identity['inputs'].get(path)
        if not expected or expected != ready.get(role + '_sha256') or sha256_file(path) != expected:
            raise ValueError('完成记录的公开图对绑定改变：' + role)
    title_digest = hashlib.sha256(public_data['source_title'].encode('utf-8')).hexdigest()
    if ready.get('source_title_sha256') != title_digest:
        raise ValueError('完成记录的公开源商品名绑定改变')
    if identity.get('pixel_optimization') is not False or identity.get('auxiliary_levels', 0) != 0 or identity.get('diffusion_calls', 0) != 0:
        raise ValueError('方法身份违反零像素及无净化条件')
    for path, digest in identity['prefix_binding']['files'].items():
        if sha256_file(path) != digest:
            raise ValueError('冻结公共前缀改变')
    candidate = ready.get('candidates', {}).get('final', {})
    if not candidate.get('sha256') or sha256_file(candidate['image']) != candidate['sha256']:
        raise ValueError('最终图片摘要改变')
    if not np.array_equal(read_rgb(candidate['image']), read_rgb(ready['generative_anchor'])):
        raise ValueError('最终图片存在像素加性残差')

    # 严格物理改动与文字保护断言验证
    prefix_root = Path(identity['prefix_binding']['root'])
    anchor_source = config.get('anchor_source', 'harmonized')
    if anchor_source == 'harmonized':
        anchor_file = prefix_root / 'color_harmonization/generative_anchor.png'
    else:
        anchor_file = prefix_root / 'adversarial.png'
    anchor_rgb = read_rgb(anchor_file)
    fg_mask = read_rgb(prefix_root / 'foreground_segmentation/source/foreground.png')[..., 0] > 0
    roi_mask = print_region_mask(anchor_rgb.shape, identity['public_definition']['print_regions'])

    if image is not None:
        if isinstance(image, (str, Path)):
            if sha256_file(image) != candidate['sha256']:
                raise ValueError('传入待验图像与完成记录摘要不符')
            final_rgb = read_rgb(image)
        elif isinstance(image, np.ndarray):
            final_rgb = image
        else:
            final_rgb = read_rgb(candidate['image'])
    else:
        final_rgb = read_rgb(candidate['image'])

    change = np.abs(final_rgb.astype(np.int16) - anchor_rgb.astype(np.int16))

    if np.any(change[roi_mask]):
        max_roi_change = int(change[roi_mask].max())
        raise AssertionError(f'印刷文字保护区保护失败：发现 {max_roi_change} 级改动 (change[roi] == 0 失败)')

    fg_max_change = int(change[fg_mask].max())
    bg_max_change = int(change[~fg_mask].max())
    if fg_max_change > config['latent_levels']:
        raise AssertionError(f'商品主体改动超过预算：{fg_max_change} > {config["latent_levels"]}')
    if bg_max_change > config['background_levels']:
        raise AssertionError(f'背景改动超过预算：{bg_max_change} > {config["background_levels"]}')

    source_rgb = read_rgb(public_data['source'])
    diff_source = np.abs(final_rgb.astype(np.float32) - source_rgb.astype(np.float32))
    mae = float(diff_source.mean())
    if mae > config['source_mae_ceiling']:
        raise AssertionError(f'全图 MAE 超出上限：{mae:.2f} > {config["source_mae_ceiling"]}')

    t_final = torch.as_tensor(final_rgb.transpose(2, 0, 1), dtype=torch.float32)[None] / 255.0
    t_source = torch.as_tensor(source_rgb.transpose(2, 0, 1), dtype=torch.float32)[None] / 255.0
    ss = float(source_ssim(t_final, t_source))
    if ss < config['source_ssim_floor']:
        raise AssertionError(f'全图 SSIM 未达门槛：{ss:.4f} < {config["source_ssim_floor"]}')

    return ready


validate_complete = validate_dual_smooth_complete


def generate_dual_smooth(
    source: Path | str,
    target: Path | str,
    output: Path | str,
    *,
    definition: dict,
    prefix_root: Path | str,
    source_public_record: dict,
    profile: str = 'dual_smooth',
    config: DualSmoothLatentConfig | None = None,
    device: str = 'cuda',
    pause_path: Path | str | None = None,
) -> tuple[Path, dict]:
    """执行商品主体曲面与背景微幅双平滑 350 步潜空间协同优化。"""
    if config is None:
        if profile not in DUAL_SMOOTH_PRESETS:
            raise ValueError(f'未知的预定义双平滑档位：{profile}')
        config = DualSmoothLatentConfig(**DUAL_SMOOTH_PRESETS[profile])
    config.validate()

    prefix, original_ready, data, teacher, parent_anchor, selected_anchor, prefix_binding = validate_prefix(
        prefix_root, definition, anchor_source=config.anchor_source)
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
        raise InterruptedError('双平滑协同试验已请求暂停')

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
        objective='equal_absolute_fullkey_joint_and_analytic_compensated_image_cosine_dual_smooth_v1',
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
        ready = validate_dual_smooth_complete(_read(root / 'ready.json'))
        del base, vae
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return Path(ready['candidates']['final']['image']), ready

    original = read_tensor(source, device)
    anchor = read_tensor(selected_anchor, device)
    oldmask = np.asarray(Image.open(relocate_io_path(prefix / 'color_harmonization/edit_mask.png')).convert('L')).copy()
    foreground = read_rgb(prefix / 'foreground_segmentation/source/foreground.png')[..., 0] > 0
    roi = print_region_mask(read_rgb(source).shape, definition['print_regions'])

    # 双余弦空间包络计算
    dist_out = distance_transform_edt(~foreground)
    if config.edge_feather_mode == 'cosine':
        t_out = np.clip(dist_out / config.edge_feather_pixels, 0.0, 1.0)
        w_bg = 0.5 * (1.0 - np.cos(np.pi * t_out)) * (~foreground)
    else:
        w_bg = np.clip(dist_out / config.edge_feather_pixels, 0.0, 1.0) * (~foreground)

    dist_in = distance_transform_edt(foreground)
    if config.edge_feather_mode == 'cosine':
        t_in = np.clip(dist_in / config.edge_feather_pixels, 0.0, 1.0)
        w_fg = 0.5 * (1.0 - np.cos(np.pi * t_in)) * foreground
    else:
        w_fg = np.clip(dist_in / config.edge_feather_pixels, 0.0, 1.0) * foreground

    # 文字边缘缓冲向外过渡
    if config.roi_feather_pixels > 0:
        dist_text = distance_transform_edt(~roi)
        t_text = np.clip(dist_text / config.roi_feather_pixels, 0.0, 1.0)
        w_text = 0.5 * (1.0 - np.cos(np.pi * t_text))
        w_fg_body = w_fg * w_text
    else:
        w_fg_body = w_fg

    # 全图空间预算分布
    level_array = (config.background_levels * w_bg + config.latent_levels * w_fg_body) * (~roi)
    max_level = float(max(config.background_levels, config.latent_levels, 1))
    macro_array = level_array / max_level
    eps_map = torch.as_tensor(level_array / 255.0, device=device, dtype=torch.float32)[None, None]
    tau_floor = max(min(config.background_levels, config.latent_levels), 1) / 255.0
    tau_map = eps_map.clamp_min(tau_floor)

    # 空间连续色度保留率图
    chroma_keep_array = (config.fg_chroma_keep * w_fg_body + config.bg_chroma_keep * w_bg).astype(np.float32)
    residual_weight = np.clip(1.0 - (w_fg_body + w_bg), 0.0, 1.0)
    chroma_keep_array = chroma_keep_array + residual_weight * config.bg_chroma_keep
    chroma_keep_map = torch.as_tensor(chroma_keep_array, device=device, dtype=torch.float32)[None, None]

    fg_bool = torch.as_tensor(foreground, device=device, dtype=torch.bool)
    bg_bool = ~fg_bool
    support_mask = torch.as_tensor((~roi).astype(np.float32), device=device)[None, None]
    fg_mask_t = torch.as_tensor((foreground & (~roi)).astype(np.float32), device=device)[None, None]
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
    branch = _DualSmoothLatentBranch(anchor, eps_map, tau_map, chroma_keep_map, support_mask, vae, config)
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

        # 主体空间平滑与 L2 惩罚
        fg_denom = max(config.latent_levels, 1) / 255.0
        fg_unit = ((image - anchor) * fg_mask_t) / fg_denom
        fg_tv = (fg_unit[..., 1:, :] - fg_unit[..., :-1, :]).abs().mean() + (fg_unit[..., :, 1:] - fg_unit[..., :, :-1]).abs().mean()
        fg_l2 = (fg_unit.square().sum() / fg_mask_t.sum().clamp_min(1.0) / 3.0)

        # 背景空间平滑、色度平滑与 L2 惩罚
        bg_denom = max(config.background_levels, 1) / 255.0
        bg_unit = ((image - anchor) * bg_mask_t) / bg_denom
        bg_tv = (bg_unit[..., 1:, :] - bg_unit[..., :-1, :]).abs().mean() + (bg_unit[..., :, 1:] - bg_unit[..., :, :-1]).abs().mean()
        bg_chroma = bg_unit - bg_unit.mean(dim=1, keepdim=True)
        bg_chroma_tv = (bg_chroma[..., 1:, :] - bg_chroma[..., :-1, :]).abs().mean() + (bg_chroma[..., :, 1:] - bg_chroma[..., :, :-1]).abs().mean()
        bg_l2 = (bg_unit.square().sum() / bg_mask_t.sum().clamp_min(1.0) / 3.0)

        overflow = (F.relu(raw - 1.0).square().mean() + F.relu(-raw).square().mean()) if raw is not None else (image.sum() * 0.0)

        penalty = (config.fg_l2_weight * fg_l2
                   + config.fg_tv_weight * fg_tv
                   + config.bg_l2_weight * bg_l2
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
            raise InterruptedError('双平滑协同试验已保存细化断点')
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
            raise FloatingPointError('双平滑试验潜变量梯度非有限')
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

    final_rgb, anchor_rgb = read_rgb(final), read_rgb(selected_anchor)
    change = np.abs(final_rgb.astype(np.int16) - anchor_rgb.astype(np.int16))
    if (np.any(change[roi])
            or int(change[foreground].max()) > config.latent_levels
            or int(change[~foreground].max()) > config.background_levels):
        raise AssertionError('商品主体预算、背景预算或印刷区域保护失败')

    metadata = dict(status='completed', config=asdict(config), history=history,
        steps_completed=step_done, best=best_row, final_check=final_check,
        accepted_updates=accepted, rejected_updates=rejected,
        latent_max_levels=int(change.max()), latent_mae_levels=float(change.mean()),
        fg_latent_max_levels=int(change[foreground].max()), fg_latent_mae_levels=float(change[foreground].mean()),
        bg_latent_max_levels=int(change[~foreground].max()), bg_latent_mae_levels=float(change[~foreground].mean()),
        auxiliary_max_levels=0, auxiliary_mae_levels=0.0, pixel_optimization=False,
        no_pixel_optimization=True, diffusion_calls=0, printed_region_fixed_to_harmonized_anchor=True,
        halo_hard_gate=False, halo_soft_weight=config.halo_soft_weight,
        mask_nonzero_pixels=int((macro_array > 0).sum()),
        mask_mean=float(macro_array.mean()), full_foreground_mask=False,
        foreground_limited=True, background_only_optimization=False,
        dual_smooth_body_and_background=True,
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
    validate_dual_smooth_complete(ready, final)
    write_json(root / 'ready.json', ready)
    save('completed')
    del base, vae, surrogate, branch, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return final, ready
