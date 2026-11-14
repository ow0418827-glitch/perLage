"""仅用于本次完整试验的主色统计、印刷核心掩码及颜色融合。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from skimage.color import rgb2lab
import torch

import prototype_v1 as palette_impl
import prototype_v2 as material_impl

RUN: Path | None = None


def rgb(path):
    return np.asarray(Image.open(path).convert('RGB')).copy()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def dominant_statistics(target, target_foreground):
    colors, labs, valid = palette_impl.palette(rgb(target), target_foreground)
    centers = np.array([c['lab'] for c in colors])
    distances = ((labs[..., None, :] - centers) * np.array([.15, 1., 1.])) ** 2
    body = valid & (np.argmin(distances.sum(-1), axis=-1) == 0)
    selected = colors[0]
    return dict(color_scheme='statistics', algorithm='foreground_merged_dominant_cluster',
                rgb=selected['rgb'], lab_mean=selected['lab'], palette=colors,
                target_foreground_pixels=int(valid.sum()), body_pixels=int(body.sum()),
                largest_cluster_fraction=selected['fraction'],
                covariance_diagnostic_only=True,
                lab_covariance=np.cov(labs[valid].T).tolist(),
                private_gallery_used=False, formal_queries_used=False), body


def reviewed_surface_masks(source, regions, output, config):
    if RUN is None:
        raise RuntimeError('试验运行目录尚未绑定')
    image = rgb(source)
    foreground = rgb(RUN / 'foreground_segmentation/source/foreground.png')[..., 0] > 0
    # 印刷候选只依赖源图及源分割，不以目标或正式结果决定保护区域。
    context = material_impl.prepare(image, foreground, image, foreground, regions)
    protected = context['protected']
    edit = np.rint(np.minimum(distance_transform_edt(~protected) / config.feather_pixels, 1.) * 255).astype(np.uint8)
    roi = np.zeros(protected.shape, bool)
    h, w = roi.shape
    for x0, y0, x1, y1 in regions:
        roi[round(y0*h):round(y1*h), round(x0*w):round(x1*w)] = True
    identity = dict(source_sha256=sha(source), regions=regions,
                    source_foreground_sha256=sha(RUN / 'foreground_segmentation/source/foreground.png'),
                    implementation_sha256=sha(__file__),
                    material_implementation_sha256=sha(Path(__file__).with_name('prototype_v2.py')),
                    palette_implementation_sha256=sha(Path(__file__).with_name('prototype_v1.py')),
                    config=dict(feather_pixels=config.feather_pixels), selection_uses_source_only=True,
                    content_semantics_certified=False)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    ip = output / 'identity.json'
    if ip.exists() and json.loads(ip.read_text(encoding='utf-8')) != identity:
        raise ValueError('新印刷核心掩码身份改变')
    files = {}
    for name, array in dict(edit=edit, ink=protected, core=protected, legacy_region=roi).items():
        path = output / (name + '.png')
        Image.fromarray(array if name == 'edit' else array.astype(np.uint8)*255).save(path)
        files[name] = dict(path=str(path.resolve()), sha256=sha(path))
    record = dict(identity=identity, files=files, legacy_protected_pixels=int(roi.sum()),
                  protected_ink_pixels=int(protected.sum()), protected_core_pixels=int(protected.sum()),
                  released_legacy_pixels=int((roi & ~protected).sum()),
                  source_derived_candidate_mask=True, requires_manual_content_and_logo_review=True,
                  caveat='Public source material separation; exact protected pixels do not certify all content semantics.')
    write(ip, identity)
    write(output / 'mask_record.json', record)
    overlay = image.copy()
    overlay[protected] = np.rint(.5*image[protected] + .5*np.array([230, 30, 50])).astype(np.uint8)
    Image.fromarray(overlay).save(output / 'protected_overlay.png')
    Image.fromarray(context['material'].astype(np.uint8)*255).save(output / 'material.png')
    return record


def dominant_fuse(data, initial, foreground, statistics, regions, root, config, base):
    from src.vision._natural.core import source_ssim
    from src.vision._natural.colour import prepare_print_context, harmonize_statistics, HarmonizationConfig, halo_error
    from src.vision.image_utils import read_tensor
    root = Path(root)
    output = root / 'color_harmonization'
    output.mkdir(parents=True, exist_ok=True)
    source = rgb(data['source'])
    target = rgb(data['target'])
    target_foreground = rgb(root / 'foreground_segmentation/target/foreground.png')[..., 0] > 0
    context = material_impl.prepare(source, foreground, target, target_foreground, regions)
    if context['target_palette'][0]['rgb'] != statistics['rgb']:
        raise ValueError('主色统计与融合选择不一致')
    anchor = rgb(initial['candidates']['generation_only']['image'])
    old_final = rgb(initial['candidates']['natural_final']['image'])
    mask = np.asarray(Image.open(initial['mask_path']).convert('L')).copy()
    mask[context['protected']] = 0
    legacy = prepare_print_context(source, anchor, mask, foreground, regions)
    if legacy['halo_applicable']:
        mask[legacy['opaque']] = 0
    residual = old_final.astype(np.int16) - anchor.astype(np.int16)
    if np.abs(residual).max() > 4:
        raise ValueError('完整初始化残差超过原有四级预算')
    residual[mask == 0] = 0
    context['input_anchor'] = anchor
    context['input_anchor_lab'] = rgb2lab(anchor / 255.)
    context['anchor_base_lab'] = context['input_anchor_lab'][context['material']].mean(0)
    original = read_tensor(data['source'], base.surrogate.device)
    parent_halo = halo_error(legacy, old_final)
    trials, chosen = [], None
    with torch.no_grad():
        for index in range(32, 0, -1):
            fraction = index / 32.
            colored = material_impl.render(context, fraction, lightness_mix=.5, chroma_scale=1.)
            if legacy['halo_applicable']:
                ring = prepare_print_context(source, colored, mask, foreground, regions)
                colored = harmonize_statistics(ring, mask, statistics['rgb'], HarmonizationConfig(**config.harmonization))
            colored[context['protected']] = source[context['protected']]
            colored[~foreground] = anchor[~foreground]
            final = np.clip(colored.astype(np.int16) + residual, 0, 255).astype(np.uint8)
            x = torch.as_tensor(final.copy(), device=base.surrogate.device, dtype=torch.float32).permute(2,0,1)[None]/255
            ss = float(source_ssim(x, original))
            mae = float((x-original).abs().mean()*255)
            halo = halo_error(legacy, final)
            protected_max = int(np.abs(final.astype(int)-source.astype(int))[context['protected']].max()) if context['protected'].any() else 0
            accepted = ss >= config.latent['source_ssim_floor']+1e-5 and mae <= config.latent['source_mae_ceiling'] and halo <= parent_halo*.5+1e-8 and protected_max == 0
            trial = dict(fraction=fraction, source_ssim=ss, source_mae_levels=mae,
                         halo_error=halo, parent_halo_error=parent_halo,
                         protected_max_difference=protected_max, accepted=accepted)
            trials.append(trial)
            if accepted:
                chosen = colored, final
                break
    if chosen is None:
        write(output / 'rejection.json', dict(trials=trials, target_rgb=statistics['rgb']))
        raise ValueError('所有非零主色融合强度均未通过原有质量及印刷光晕限制')
    Image.fromarray(chosen[0]).save(output / 'generative_anchor.png')
    Image.fromarray(chosen[1]).save(output / 'initial.png')
    Image.fromarray(mask).save(output / 'edit_mask.png')
    record = dict(status='completed', color_scheme='statistics', selected_rgb=statistics['rgb'],
                  algorithm='dominant_material_color_before_full_latent_refinement',
                  chosen=trials[-1], trials=trials, halo_applicable=legacy['halo_applicable'],
                  foreground_limited=True, printed_core_preserved=True,
                  auxiliary_residual_fixed_for_latent_stage=True,
                  selection='largest_nonzero_quality_feasible_fraction',
                  intermediate_similarity_used_to_abort_or_select_fraction=False,
                  ordinary_teacher='post_color_initial_image', halo_teacher='pre_color_complete_initialization',
                  private_gallery_used=False, formal_queries_used=False)
    write(output / 'metadata.json', record)
    return output, record
