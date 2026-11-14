import json
from pathlib import Path

p = Path("output/experiment_base_groups9_20261109/manifest.json")
d = json.loads(p.read_text(encoding="utf-8"))
d["shared_memory"]["query_index_cache"] = {
    "cache_file": "shared_memory/query_index_cache/formal_queries_vit_b32.json",
    "cache_sha256": "3c31ccd3dd74a0223e3aa7bdfd846b42eae4be2c19107bd503a7aa920025c95d",
    "query_count": 180,
    "zero_gpu_retrieval_supported": True,
}
p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
print("manifest.json updated successfully with query_index_cache!")
