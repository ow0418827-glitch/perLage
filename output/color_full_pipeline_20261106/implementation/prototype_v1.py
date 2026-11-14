"""公开前景主色簇与底材改色的第一版独立候选；尚未注册到主程序。"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, binary_propagation, gaussian_filter
from skimage.color import rgb2lab, lab2rgb
from sklearn.cluster import KMeans


def palette(image, foreground):
    image = np.asarray(image, dtype=np.uint8)
    fg = np.asarray(foreground, dtype=bool)
    border = np.concatenate([image[0], image[-1], image[:, 0], image[:, -1]])
    bg = np.median(border, axis=0)
    possible = np.linalg.norm(image.astype(float) - bg, axis=-1) <= 10
    seeds = np.zeros_like(possible)
    seeds[[0, -1], :] = possible[[0, -1], :]
    seeds[:, [0, -1]] = possible[:, [0, -1]]
    connected = binary_propagation(seeds, mask=possible)
    valid = binary_erosion(fg & ~connected, iterations=1)
    if valid.sum() < 100:
        raise ValueError('有效商品前景不足')
    labs = rgb2lab(image / 255.)
    values = labs[valid]
    features = values * np.array([.15, 1., 1.])
    sample = features[np.linspace(0, len(features) - 1, min(len(features), 12000)).astype(int)]
    km = KMeans(n_clusters=min(6, len(np.unique(sample.round(1), axis=0))),
                random_state=42, n_init=1, max_iter=60).fit(sample)
    assignments = np.argmin(((features[:, None] - km.cluster_centers_[None]) ** 2).sum(-1), axis=1)
    centers = []
    for i in range(len(km.cluster_centers_)):
        v = values[assignments == i]
        if len(v):
            centers.append({'lab': v.mean(0), 'count': len(v), 'members': [i]})
    # 合并色相、饱和度接近而仅受光照影响的颜色簇，防止主体被明暗切碎。
    while True:
        distances = [(float(np.linalg.norm(a['lab'][1:] - b['lab'][1:])), i, j)
                     for i, a in enumerate(centers) for j, b in enumerate(centers) if j > i]
        if not distances or min(distances)[0] > 8.:
            break
        _, i, j = min(distances)
        a, b = centers[i], centers[j]
        total = a['count'] + b['count']
        centers[i] = {'lab': (a['lab'] * a['count'] + b['lab'] * b['count']) / total,
                      'count': total, 'members': a['members'] + b['members']}
        centers.pop(j)
    centers.sort(key=lambda x: (-x['count'], tuple(x['lab'])))
    records = [{'lab': c['lab'].tolist(), 'rgb': np.rint(lab2rgb(c['lab'][None, None])[0, 0] * 255).astype(int).tolist(),
                'pixels': int(c['count']), 'fraction': c['count'] / len(values)} for c in centers]
    return records, labs, valid


def prepare(source, source_foreground, target, target_foreground, regions):
    src_palette, source_lab, valid = palette(source, source_foreground)
    target_palette, _, target_valid = palette(target, target_foreground)
    base = np.asarray(src_palette[0]['lab'])
    fg = np.asarray(source_foreground, bool)
    roi = np.zeros(fg.shape, bool)
    h, w = roi.shape
    for x0, y0, x1, y1 in regions:
        roi[round(y0 * h):round(y1 * h), round(x0 * w):round(x1 * w)] = True
    material = (np.linalg.norm(source_lab[..., 1:] - base[1:], axis=-1) <= 12.)
    material &= source_lab[..., 0] >= base[0] - 28.
    material &= fg
    ink = roi & fg & ~material
    protected = binary_dilation(ink, iterations=1) & fg
    material &= ~protected
    # 从边界内侧羽化，保护文字核心和轮廓；不在背景上生成人工纹理。
    from scipy.ndimage import distance_transform_edt
    weight = np.minimum(distance_transform_edt(material) / 2., 1.)
    return dict(source=np.asarray(source).copy(), source_lab=source_lab, source_base=base,
                foreground=fg, material=material, protected=protected, weight=weight,
                source_palette=src_palette, target_palette=target_palette,
                target_valid_pixels=int(target_valid.sum()))


def render(context, alpha, *, palette_index=0, lightness_mix=.5, chroma_scale=1.):
    src = context['source_lab']
    base = context['source_base']
    color = np.asarray(context['target_palette'][palette_index]['lab'])
    desired = src.copy()
    desired[..., 0] += lightness_mix * (color[0] - base[0])
    desired[..., 1:] = color[1:] * chroma_scale + .3 * (src[..., 1:] - base[1:])
    mix = alpha * context['weight'][..., None]
    edited = src + mix * (desired - src)
    edited[..., 0] = edited[..., 0].clip(0, 100)
    output = np.rint(np.clip(lab2rgb(edited), 0, 1) * 255).astype(np.uint8)
    output[context['weight'] == 0] = context['source'][context['weight'] == 0]
    output[context['protected']] = context['source'][context['protected']]
    return output


def candidates(context):
    for alpha in np.linspace(0, 1, 17):
        yield {'alpha': float(alpha), 'palette_index': 0, 'lightness_mix': .5, 'chroma_scale': 1.}, render(context, alpha)
