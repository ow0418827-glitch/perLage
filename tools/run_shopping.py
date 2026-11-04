"""未来正式购物评估入口；默认关闭，不更改停止购买标记。"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.paths import OUTPUT, CONFIGS
from src.experiments.shopping_evaluation import run_shopping_evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--method', choices=('lucid','PerLAGE'), required=True)
    parser.add_argument('--model-config', type=Path, default=CONFIGS / 'llm.local.json')
    parser.add_argument('--enable-shopping', action='store_true')
    args = parser.parse_args()
    root = args.run_dir.resolve()
    if not root.is_relative_to(OUTPUT.resolve()):
        parser.error('只能使用新项目运行目录')
    run_shopping_evaluation(root/'methods'/args.method, model_config=args.model_config,
                            allow_shopping=args.enable_shopping)


if __name__ == '__main__':
    main()
