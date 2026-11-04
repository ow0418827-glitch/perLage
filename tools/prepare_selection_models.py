"""显式准备新增公开选取代理；优先使用中国大陆镜像。"""
import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint', default='https://hf-mirror.com')
    args = parser.parse_args()
    os.environ['HF_ENDPOINT'] = args.endpoint
    os.environ.pop('HF_HUB_OFFLINE', None)
    os.environ.pop('TRANSFORMERS_OFFLINE', None)
    from huggingface_hub import snapshot_download
    import torch
    from src.paths import MODEL_ROOT
    snapshot_download('facebook/dinov2-small', local_dir=MODEL_ROOT / 'dinov2-small',
                      allow_patterns=['config.json', 'preprocessor_config.json', 'model.safetensors'])
    snapshot_download('sentence-transformers/all-MiniLM-L6-v2',
                      allow_patterns=['*.json', '*.txt', 'model.safetensors', '1_Pooling/*'])
    # 此主干使用视觉库公布的官方原始权重及其内容校验；不替换成随机权重。
    from src.vision.image_utils import sha256_file
    checkpoint = Path(torch.hub.get_dir()) / 'checkpoints/alexnet-owt-7be5be79.pth'
    if not checkpoint.is_file():
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        from torchvision.models import AlexNet_Weights
        torch.hub.download_url_to_file(AlexNet_Weights.IMAGENET1K_V1.url,
                                      str(checkpoint), hash_prefix='7be5be79')
    print('新增公开代理准备完成；感知主干摘要：' + sha256_file(checkpoint))


if __name__ == '__main__':
    main()
