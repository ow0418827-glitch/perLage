"""准备完整流程的公开模型；不读取任何记忆或实验结果。"""
import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope', choices=('generation','encoders','all'), default='all')
    parser.add_argument('--endpoint', default='https://hf-mirror.com')
    args = parser.parse_args()
    os.environ['HF_ENDPOINT'] = args.endpoint
    os.environ.pop('HF_HUB_OFFLINE', None)
    from huggingface_hub import snapshot_download
    from src.paths import MODEL_ROOT, CONFIGS
    from src.vision.audit import read_json, save_json, model_files
    if args.scope in ('generation','all'):
        downloads = [
            ('facebook/sam2.1-hiera-tiny','sam2.1-hiera-tiny-source', ['*.json','model.safetensors','README.md','LICENSE*']),
            ('stable-diffusion-v1-5/stable-diffusion-v1-5','stable-diffusion-v1-5',
             ['model_index.json','README.md','LICENSE*','scheduler/*','tokenizer/*','unet/config.json',
              'vae/config.json','text_encoder/config.json','unet/*fp16.safetensors','vae/*fp16.safetensors','text_encoder/*fp16.safetensors']),
            ('timbrooks/instruct-pix2pix','instruct-pix2pix-editor',
             ['model_index.json','README.md','LICENSE*','scheduler/*','unet/config.json','unet/*fp16.safetensors']),
        ]
        for repo, name, patterns in downloads:
            root = MODEL_ROOT / name
            snapshot_download(repo, local_dir=root, allow_patterns=patterns)
            save_json(root/'download_manifest.json', dict(source=repo, endpoint=args.endpoint, model=model_files(root)))
            print('已准备公开模型：' + name, flush=True)
    if args.scope in ('encoders','all'):
        specs = read_json(CONFIGS/'optimization.json')['surrogates']
        specs.append(read_json(CONFIGS/'evaluator.example.json')['surrogate'])
        for spec in specs:
            snapshot_download(spec['hf_repo'], allow_patterns=['open_clip_model.safetensors','open_clip_pytorch_model.safetensors','open_clip_config.json','*.json','*.txt'])
            print('已准备公开编码器：' + spec['name'] + '/' + spec['pretrained'], flush=True)


if __name__ == '__main__':
    main()
