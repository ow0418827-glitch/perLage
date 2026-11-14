"""用户授权：仅本次洗发水临时放宽光晕上限，其余融合代码保持原样。"""
from pathlib import Path
import numpy as np
from PIL import Image
from skimage.color import rgb2lab
import torch
import trial_components as original
material_impl=original.material_impl
rgb,sha,write=original.rgb,original.sha,original.write
dominant_statistics=original.dominant_statistics
RUN=None
def reviewed_surface_masks(source,regions,output,config):
    original.RUN=RUN
    return original.reviewed_surface_masks(source,regions,output,config)

def dominant_fuse(data, initial, foreground, statistics, regions, root, config, base):
    if RUN is None or RUN.parent.name != '07_shampoo' or RUN.parent.parent.name != 'dominant':
        raise ValueError('临时放宽仅允许本次洗发水新方案')
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
            accepted = ss >= config.latent['source_ssim_floor']+1e-5 and mae <= config.latent['source_mae_ceiling'] and halo <= parent_halo*2.0+1e-8 and protected_max == 0
            trial = dict(fraction=fraction, source_ssim=ss, source_mae_levels=mae,
                         halo_error=halo, parent_halo_error=parent_halo,
                         protected_max_difference=protected_max, accepted=accepted, original_halo_limit=parent_halo*.5+1e-8, temporary_halo_limit=parent_halo*2.0+1e-8)
            trials.append(trial)
            if accepted:
                chosen = colored, final
                break
    if chosen is None:
        write(output / 'rejection.json', dict(trials=trials, target_rgb=statistics['rgb']))
        raise ValueError('洗发水在本次临时光晕上限下仍无可行的非零融合强度')
    Image.fromarray(chosen[0]).save(output / 'generative_anchor.png')
    Image.fromarray(chosen[1]).save(output / 'initial.png')
    Image.fromarray(mask).save(output / 'edit_mask.png')
    record = dict(status='completed', color_scheme='statistics', selected_rgb=statistics['rgb'],
                  algorithm='dominant_material_color_before_full_latent_refinement', temporary_halo_multiplier=2.0, original_halo_multiplier=0.5, scope='07_shampoo_only_experimental_override', main_program_defaults_changed=False,
                  chosen=trials[-1], trials=trials, halo_applicable=legacy['halo_applicable'],
                  foreground_limited=True, printed_core_preserved=True,
                  auxiliary_residual_fixed_for_latent_stage=True,
                  selection='largest_nonzero_quality_feasible_fraction',
                  intermediate_similarity_used_to_abort_or_select_fraction=False,
                  ordinary_teacher='post_color_initial_image', halo_teacher='pre_color_complete_initialization',
                  private_gallery_used=False, formal_queries_used=False)
    write(output / 'metadata.json', record)
    return output, record
