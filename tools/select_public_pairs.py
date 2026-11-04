"""按固定公开目标，从商城原始商品数据准备并选取源商品图。"""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.experiments.public_pair_selection import main

if __name__=='__main__':raise SystemExit(main())
