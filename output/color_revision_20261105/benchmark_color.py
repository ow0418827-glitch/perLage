"""以真实本地编码器对公开颜色子步骤做对照，不读取私有记忆或正式查询。"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
sys.path.insert(0, str(PROJECT))
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from src.vision.dual_rag_bridge import make_encoder
from src.vision.joint_encoder import ChunkedClipEncoder
from src.vision._natural.core import source_ssim


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def rgb(path):
    return np.asarray(Image.open(path).convert('RGB')).copy()


def tensor(array, device):
    return torch.from_numpy(array.copy()).to(device=device, dtype=torch.float32).permute(2, 0, 1)[None] / 255.


def score_encoder(base, array, source_title, queries, target):
    enc = ChunkedClipEncoder(base)
    title = torch.tensor(enc.encode_text(source_title), device=base.device)[None]
    with torch.no_grad():
        target_feature = base.surrogate.encode_image(tensor(target, base.device)).global_embedding
        query = torch.cat([F.normalize(target_feature + torch.tensor(enc.encode_text(q), device=base.device)[None], dim=-1) for q in queries])
        feature = base.surrogate.encode_image(tensor(array, base.device)).global_embedding
    return {'image_cosine': float((feature @ target_feature.T).item()),
            'joint_mean': float((F.normalize(feature + title, dim=-1) @ query.T).mean())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=int, choices=[1, 2], required=True)
    args = parser.parse_args()
    version = f'v{args.version}'
    spec = importlib.util.spec_from_file_location('color_candidate', ROOT / f'prototype_{version}.py')
    proto = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(proto)
    plan = read(ROOT / 'PLAN.json')
    out = ROOT / version
    out.mkdir(parents=True, exist_ok=True)
    device = 'cuda'
    torch.set_num_threads(4)
    started = time.perf_counter()
    base = make_encoder(device=device)
    enc = ChunkedClipEncoder(base)
    rows = []
    for case in plan['cases']:
        key = case['key']
        d = out / key
        d.mkdir(exist_ok=True)
        source, target = rgb(case['source']), rgb(case['target'])
        foreground = rgb(case['source_foreground'])[..., 0] > 0
        target_fg = rgb(case['target_foreground'])[..., 0] > 0
        context = proto.prepare(source, foreground, target, target_fg, case['print_regions'])
        context['input_anchor'] = rgb(case['baseline_color']) if args.version == 2 else source.copy()
        for name in ['material', 'protected']:
            Image.fromarray(context[name].astype(np.uint8) * 255).save(d / (name + '.png'))
        save(d / 'palette.json', {'source': context['source_palette'], 'target': context['target_palette']})
        original = tensor(source, device)
        title = torch.tensor(enc.encode_text(case['source_title']), device=device)[None]
        with torch.no_grad():
            target_feature = base.surrogate.encode_image(tensor(target, device)).global_embedding
            queries = torch.cat([F.normalize(target_feature + torch.tensor(enc.encode_text(q), device=device)[None], dim=-1) for q in plan['generic_queries']])
        trials, best, best_array = [], None, None
        for params, array in proto.candidates(context):
            x = tensor(array, device)
            with torch.no_grad():
                quality = float(source_ssim(x, original))
                mae = float((x - original).abs().mean() * 255)
                feature = base.surrogate.encode_image(x).global_embedding
                cosine = float((feature @ target_feature.T).item())
                joint = float((F.normalize(feature + title, dim=-1) @ queries.T).mean())
            valid = quality >= .85 and mae <= 12. and np.array_equal(array[context['protected']], source[context['protected']])
            row = dict(params=params, source_ssim=quality, source_mae=mae, image_cosine=cosine,
                       joint_mean=joint, feasible=bool(valid), utility=cosine + .5 * joint)
            trials.append(row)
            if valid and (best is None or row['utility'] > best['utility']):
                best, best_array = row, array
        if best is None:
            raise ValueError(f'{key}: 没有通过质量约束的候选')
        Image.fromarray(best_array).save(d / 'candidate.png')
        baseline = rgb(case['baseline_color'])
        reference = score_encoder(base, baseline, case['source_title'], plan['generic_queries'], target)
        source_score = score_encoder(base, source, case['source_title'], plan['generic_queries'], target)
        mask = context['material']
        from skimage.color import rgb2lab
        target_ab = np.array(context['target_palette'][0]['lab'])[1:]
        old_distance = float(np.linalg.norm(rgb2lab(baseline / 255.)[..., 1:][mask] - target_ab, axis=-1).mean())
        new_distance = float(np.linalg.norm(rgb2lab(best_array / 255.)[..., 1:][mask] - target_ab, axis=-1).mean())
        result = dict(group=key, version=version, scope='color_stage_only', baseline=reference,
             source=source_score, candidate=best, image_gain=best['image_cosine'] - reference['image_cosine'],
             joint_gain=best['joint_mean'] - reference['joint_mean'], source_palette=context['source_palette'],
             target_palette=context['target_palette'], material_pixels=int(mask.sum()),
             protected_pixels=int(context['protected'].sum()),
             protected_max_difference=int(np.abs(best_array.astype(int)-source.astype(int))[context['protected']].max()) if context['protected'].any() else 0,
             background_max_difference=int(np.abs(best_array.astype(int)-context['input_anchor'].astype(int))[~foreground].max()),
             background_reference='fixed_color_input_anchor' if args.version==2 else 'original_source_input_anchor',
             background_difference_from_original_source=int(np.abs(best_array.astype(int)-source.astype(int))[~foreground].max()),
             color_distance_before=old_distance, color_distance_after=new_distance,
             color_distance_reduction=1-new_distance/max(old_distance, 1e-8), trials=trials)
        save(d / 'result.json', result)
        rows.append(result)
        print(json.dumps({k: result[k] for k in ['group', 'image_gain', 'joint_gain', 'color_distance_reduction']}, ensure_ascii=False), flush=True)
    primary_snapshot = base.snapshot
    del enc, base
    gc.collect()
    torch.cuda.empty_cache()
    secondary = make_encoder(surrogate=1, device=device)
    for row, case in zip(rows, plan['cases']):
        a = score_encoder(secondary, rgb(out / case['key'] / 'candidate.png'), case['source_title'], plan['generic_queries'], rgb(case['target']))
        b = score_encoder(secondary, rgb(case['baseline_color']), case['source_title'], plan['generic_queries'], rgb(case['target']))
        row['secondary'] = dict(candidate=a, baseline=b, image_gain=a['image_cosine']-b['image_cosine'], joint_gain=a['joint_mean']-b['joint_mean'])
        save(out / case['key'] / 'result.json', row)
    required = [r for r in rows if r['group'] in plan['promotion_gate']['primary_required_cases']]
    gate = plan['promotion_gate']
    checks = dict(mean_image_gain=float(np.mean([r['image_gain'] for r in required])) >= gate['mean_image_cosine_gain_at_least'],
        each_image_gain=all(r['image_gain'] >= gate['each_required_image_cosine_gain_at_least'] for r in required),
        each_joint_gain=all(r['joint_gain'] >= gate['each_required_joint_gain_at_least'] for r in required),
        chair_regression=all(min(r['image_gain'], r['joint_gain']) >= -gate['chair_image_or_joint_regression_at_most'] for r in rows if r['group']=='02_chair'),
        secondary_mean=float(np.mean([r['secondary']['image_gain'] for r in required])) >= gate['secondary_encoder_mean_gain_at_least'],
        secondary_no_regression=all(r['secondary']['image_gain'] >= -gate['secondary_encoder_case_regression_at_most'] for r in rows),
        color_distance=all(r['color_distance_reduction'] >= gate['material_color_distance_reduction_at_least'] for r in required),
        protected_and_background=all(r['protected_max_difference']==0 and r['background_max_difference']==0 for r in rows))
    result = dict(version=version, completed=True, scope='public_color_stage_only',
                  checks=checks, numerical_gate_passed=all(checks.values()), manual_visual_review_pending=True,
                  primary_encoder=primary_snapshot, secondary_encoder=secondary.snapshot,
                  elapsed_seconds=time.perf_counter()-started, rows=rows,
                  private_bank_used=False, formal_queries_used=False, shopping_runs=0)
    save(out / 'RESULT.json', result)
    print(json.dumps({'version':version, 'checks':checks, 'elapsed_seconds':result['elapsed_seconds']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
