"""显式准备新增公开选取代理；优先使用中国大陆镜像。"""
import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.paths import ProjectPath as Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint', default='https://hf-mirror.com')
    args = parser.parse_args()
    os.environ['HF_ENDPOINT'] = args.endpoint
    os.environ.pop('HF_HUB_OFFLINE', None)
    os.environ.pop('TRANSFORMERS_OFFLINE', None)
    from huggingface_hub import snapshot_download
    from src.paths import MODEL_ROOT
    snapshot_download('facebook/dinov2-small', local_dir=MODEL_ROOT / 'dinov2-small',
                      allow_patterns=['config.json', 'preprocessor_config.json', 'model.safetensors'])
    print('轻量选图所需的单一冻结视觉模型已准备。正式生成所需模型仍由生成阶段准备。')


if __name__ == '__main__':
    main()
