"""导出诊断图、核实原方法未变，并将未采用候选归档。"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


plan = read(ROOT / 'PLAN.json')
spec = importlib.util.spec_from_file_location('color_v2_export', ROOT / 'prototype_v2.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
for case in plan['cases']:
    def image(path):
        return np.asarray(Image.open(path).convert('RGB')).copy()
    context = module.prepare(image(case['source']), image(case['source_foreground'])[..., 0] > 0,
                             image(case['target']), image(case['target_foreground'])[..., 0] > 0,
                             case['print_regions'])
    context['input_anchor'] = image(case['baseline_color'])
    context['input_anchor_lab'] = module.rgb2lab(context['input_anchor'] / 255.)
    context['anchor_base_lab'] = context['input_anchor_lab'][context['material']].mean(0)
    forced = module.render(context, 1., lightness_mix=.5, chroma_scale=1.)
    Image.fromarray(forced).save(ROOT / 'v2' / case['key'] / 'forced_full_color.png')

checks = {}
for path in (ROOT / 'original_implementation').rglob('*.py'):
    current = PROJECT / path.relative_to(ROOT / 'original_implementation')
    checks[str(current)] = {'backup_sha256': sha(path), 'current_sha256': sha(current)}
assert all(v['backup_sha256'] == v['current_sha256'] for v in checks.values())
assert all(sha(Path(p)) == s for p, s in plan['original_input_sha256'].items())
save(ROOT / 'PRESERVATION.json', dict(active_implementation_unchanged=True,
     original_public_inputs_unchanged=True, method_files=checks))

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font = FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    result = read(ROOT / 'v2/RESULT.json')
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout='constrained')
    for ax, key, title in zip(axes, ['07_shampoo', '09_toothbrush'], ['洗发水组', '牙刷组']):
        row = next(r for r in result['rows'] if r['group'] == key)
        trials = [r for r in row['trials'] if r['params']['lightness_mix'] == .5 and r['params']['chroma_scale'] == 1.]
        ax.plot([r['params']['alpha'] for r in trials], [r['image_cosine'] for r in trials],
                marker='o', markersize=3, label='第二版向目标主色改色')
        ax.axhline(row['baseline']['image_cosine'], color='#c34e44', linestyle='--', label='原颜色阶段输出')
        ax.set_xlabel('改色强度', fontproperties=font)
        ax.set_ylabel('与目标图的图像余弦相似度', fontproperties=font)
        ax.set_title(title, fontproperties=font)
        ax.grid(alpha=.2)
        ax.legend(prop=font)
    figure.savefig(ROOT / 'COLOR_VS_SIMILARITY.png', dpi=180)
    plt.close(figure)
except ImportError:
    pass

v1, v2 = [read(ROOT / v / 'RESULT.json') for v in ['v1', 'v2']]
assert not v1['numerical_gate_passed'] and not v2['numerical_gate_passed']
save(ROOT / 'DECISION.json', dict(decision='retain_original_method', candidate_versions_tested=2,
     reason_zh='第一版相似度下降；第二版最佳改色强度对洗发水和牙刷均为零，增益微小且没有缩小主体色差。',
     main_program_modified=False, original_method_retained=True, scope='public_color_stage_only',
     full_new_method_generated=False, private_bank_access=False, formal_query_access=False, shopping_runs=0,
     parameter_combinations=sum(len(row['trials']) for v in [v1, v2] for row in v['rows']),
     total_automatic_benchmark_seconds=v1['elapsed_seconds'] + v2['elapsed_seconds']))

archive = ROOT / 'UNPROMOTED_COLOR_METHODS.zip'
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as output:
    for path in sorted(ROOT.rglob('*')):
        if path.is_file() and path not in [archive, ROOT / 'BACKUP_MANIFEST.json'] and '__pycache__' not in path.parts:
            output.write(path, path.relative_to(ROOT).as_posix())
with zipfile.ZipFile(archive) as output:
    assert output.testzip() is None
save(ROOT / 'BACKUP_MANIFEST.json', dict(path=str(archive), sha256=sha(archive), bytes=archive.stat().st_size,
     includes=['两版候选实现', '原实现备份', '公开输入及固定方案', '两轮完整数值及试改色图片', '诊断与决策']))
print('两版未达到替换标准；原方法保留。候选归档完整，原程序及原始公开素材摘要一致。')
