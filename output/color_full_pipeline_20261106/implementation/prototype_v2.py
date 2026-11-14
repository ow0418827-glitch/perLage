"""第二版：按源商品标注区域识别底材，保留固定初始化的细节后搜索颜色参数。"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt
from skimage.color import rgb2lab, lab2rgb

spec = importlib.util.spec_from_file_location('color_v1_internal', Path(__file__).with_name('prototype_v1.py'))
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)


def prepare(source, source_foreground, target, target_foreground, regions):
    context = common.prepare(source, source_foreground, target, target_foreground, regions)
    fg = context['foreground']
    roi = np.zeros(fg.shape, bool)
    h, w = roi.shape
    for x0, y0, x1, y1 in regions:
        roi[round(y0 * h):round(y1 * h), round(x0 * w):round(x1 * w)] = True
    region_fg = fg & roi
    if region_fg.sum() >= 100:
        regional_palette, _, _ = common.palette(source, region_fg)
        base = np.asarray(regional_palette[0]['lab'])
        context['material_reference_palette'] = regional_palette
    else:
        base = context['source_base']
        context['material_reference_palette'] = context['source_palette']
    labs = context['source_lab']
    material = fg & (np.linalg.norm(labs[..., 1:] - base[1:], axis=-1) <= 14.)
    material &= labs[..., 0] >= base[0] - 28.
    ink = roi & fg & ~material
    protected = binary_dilation(ink, iterations=1) & fg
    material &= ~protected
    context.update(source_base=base, material=material, protected=protected,
                   weight=np.minimum(distance_transform_edt(material) / 2., 1.))
    return context


def render(context, alpha, *, palette_index=0, lightness_mix=.5, chroma_scale=1.):
    anchor = context['input_anchor']
    labs = context['input_anchor_lab']
    target = np.asarray(context['target_palette'][palette_index]['lab'])
    base = context['anchor_base_lab']
    shift = np.array([(target[0] - base[0]) * lightness_mix,
                      target[1] * chroma_scale - base[1], target[2] * chroma_scale - base[2]])
    shifted = labs + alpha * context['weight'][..., None] * shift
    shifted[..., 0] = np.clip(shifted[..., 0], 0, 100)
    result = np.rint(np.clip(lab2rgb(shifted), 0, 1) * 255).astype(np.uint8)
    result[context['weight'] == 0] = anchor[context['weight'] == 0]
    result[context['protected']] = context['source'][context['protected']]
    return result


def candidates(context):
    anchor = context['input_anchor']
    context['input_anchor_lab'] = rgb2lab(anchor / 255.)
    context['anchor_base_lab'] = context['input_anchor_lab'][context['material']].mean(0)
    for alpha in np.linspace(0, 1, 17):
        for lightness in [0., .5, 1.]:
            for saturation in [.75, 1., 1.25]:
                params = dict(alpha=float(alpha), palette_index=0,
                              lightness_mix=lightness, chroma_scale=saturation)
                yield params, render(context, **params)
