"""将显式授权的记忆输入合并为唯一共享池；不执行购物。"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.memory.shared_pool import build_shared_pool
from src.paths import SHARED_MEMORY


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, nargs='+', required=True)
    parser.add_argument('--kind', choices=('purchase_memory', 'simulated_gallery', 'preserve'), required=True)
    parser.add_argument('--output', type=Path, default=SHARED_MEMORY)
    args = parser.parse_args()
    pool = build_shared_pool(args.input, args.output, kind=args.kind)
    print(f'共享池已准备，共 {len(pool["entries"])} 条：{args.output.resolve()}')


if __name__ == '__main__':
    main()
