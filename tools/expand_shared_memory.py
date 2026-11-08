"""将既有共享池扩为九组各二十一条的统一全局池，不调用购物模型。"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.paths import ProjectPath as Path
from src.agent.memory.balanced_pool import expand_pool
from src.paths import SHARED_MEMORY, PRIVATE, PUBLIC


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=SHARED_MEMORY)
    parser.add_argument('--base',type=Path,default=PRIVATE/'shared_memory.json')
    parser.add_argument('--public-index',type=Path,default=PUBLIC/'public_products_v2.sqlite')
    args=parser.parse_args()
    entries=expand_pool(args.output,base_path=args.base,public_index=args.public_index)
    print('统一共享池完成，共',len(entries),'条。')
