"""Build and freeze official query vector cache for 9 groups."""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import EXPERIMENT_BASE_GROUPS9
from src.visual.query_cache import build_groups9_query_cache

if __name__ == "__main__":
    cache = build_groups9_query_cache(EXPERIMENT_BASE_GROUPS9)
    print(f"9组正式查询向量缓存构建完成，总数: {len(cache)} 条")
