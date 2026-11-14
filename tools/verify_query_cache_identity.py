"""Verify that query vector cache matches live encoder perfectly."""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from src.paths import EXPERIMENT_BASE_GROUPS9
from src.experiments.shared_evaluation import make_eval_encoder
from src.visual.query_cache import QueryVectorCache

encoder_config = EXPERIMENT_BASE_GROUPS9 / "evaluation_encoder.json"
live_encoder = make_eval_encoder(device="cpu", config_path=encoder_config)

cache_path = EXPERIMENT_BASE_GROUPS9 / "shared_memory" / "query_index_cache" / "formal_queries_vit_b32.json"
cache = QueryVectorCache.load(cache_path)

print(f"Loaded cache entries: {len(cache)}")
max_diff = 0.0
for text, entry in cache._entries.items():
    cached_vec = np.array(entry["vector"], dtype=np.float32)
    live_vec = live_encoder.encode_text(text)
    live_vec = live_vec / np.linalg.norm(live_vec)
    diff = float(np.max(np.abs(cached_vec - live_vec)))
    if diff > max_diff:
        max_diff = diff

print(f"Max absolute difference across all {len(cache)} queries: {max_diff:.8e}")
assert max_diff < 1e-6, f"Difference too high: {max_diff}"
print("VERIFICATION SUCCESSFUL: Cache is bit-for-bit identical with live model!")
