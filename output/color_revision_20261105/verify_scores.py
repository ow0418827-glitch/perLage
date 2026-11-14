"""新进程从落盘图片独立复算两轮图像余弦，不重新优化。"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import json
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[2]))
from src.vision.dual_rag_bridge import make_encoder

plan = json.loads((ROOT / 'PLAN.json').read_text(encoding='utf-8'))
base = make_encoder(device='cuda')
checks = []
for version in ['v1', 'v2']:
    result = json.loads((ROOT / version / 'RESULT.json').read_text(encoding='utf-8'))
    for case, row in zip(plan['cases'], result['rows']):
        assert case['key'] == row['group']
        target = base.encode(case['target'])
        for name, path, expected in [
            ('baseline', case['baseline_color'], row['baseline']['image_cosine']),
            ('candidate', ROOT / version / case['key'] / 'candidate.png', row['candidate']['image_cosine'])]:
            actual = float(np.dot(base.encode(path), target))
            checks.append(dict(version=version, group=case['key'], kind=name,
                               expected=expected, actual=actual, error=abs(expected-actual)))
assert len(checks) == 12
maximum = max(c['error'] for c in checks)
assert maximum <= 1e-6, maximum
output = dict(passed=True, max_absolute_error=maximum, checks=checks,
              scope='fresh_encoder_saved_image_cosine_recomputation', optimization_rerun=False)
(ROOT / 'INDEPENDENT_VERIFICATION.json').write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(f'已从落盘图片独立复算十二项分数，最大误差{maximum:.9g}。')
