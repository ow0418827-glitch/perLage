"""未来正式购物评估入口；默认关闭，不更改停止购买标记。"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.paths import ProjectPath as Path
from src.paths import OUTPUT, CONFIGS
from src.experiments.shopping_evaluation import run_shopping_evaluation
from src.visual.optimization import METHODS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--model-config', type=Path, default=CONFIGS / 'llm.local.json')
    parser.add_argument('--enable-shopping', action='store_true')
    prompts=parser.add_mutually_exclusive_group()
    prompts.add_argument('--prompt-profile',help='智能体配置中登记的附加指导配置名')
    prompts.add_argument('--prompt-template',type=Path,help='独立附加指导文件')
    args = parser.parse_args()
    root = args.run_dir.resolve()
    if not root.is_relative_to(OUTPUT.resolve()):
        parser.error('只能使用新项目运行目录')
    run_shopping_evaluation(root/'methods'/args.method, model_config=args.model_config,
                            allow_shopping=args.enable_shopping,
                            prompt_path=args.prompt_template,prompt_profile=args.prompt_profile)


if __name__ == '__main__':
    main()
