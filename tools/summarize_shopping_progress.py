import json
from pathlib import Path
from collections import Counter

p = Path("output/lucid_groups9_20261109/shopping/outcomes.json")
if not p.is_file():
    print("No outcomes file yet.")
    exit(0)

data = json.loads(p.read_text(encoding="utf-8"))
c = Counter(o["group"] for o in data)
mis = Counter(o["group"] for o in data if o["misleading_purchase"])
tgt = Counter(o["group"] for o in data if o["exact_target_purchase"])

print(f"Total episodes completed: {len(data)} / 180 ({(len(data)/180)*100:.1f}%)")
print(f"Total misleading purchases: {sum(mis.values())} ({(sum(mis.values())/len(data))*100:.1f}%)")
print(f"Total exact target purchases: {sum(tgt.values())} ({(sum(tgt.values())/len(data))*100:.1f}%)")
print("-" * 60)
for g in sorted(c.keys()):
    tot = c[g]
    m = mis[g]
    t = tgt[g]
    other = tot - m - t
    print(f"[{g:18s}] 已测: {tot:2d}/20 | 误购源商品: {m:2d} ({m/tot*100:5.1f}%) | 购入目标: {t:2d} ({t/tot*100:4.1f}%) | 其他: {other:2d}")
