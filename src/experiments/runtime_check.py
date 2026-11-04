"""离线核查当前项目真实模型和可导入组件。"""
import importlib
import json
from pathlib import Path

from src.paths import PROJECT, MODEL_ROOT, SHARED_MEMORY


def doctor(device='cuda'):
    import torch
    packages = {}
    for name in ('torch', 'transformers', 'diffusers', 'accelerate', 'open_clip', 'sentence_transformers', 'lpips', 'skimage'):
        try:
            module = importlib.import_module(name)
            packages[name] = dict(available=True, version=getattr(module, '__version__', None))
        except ImportError as error:
            packages[name] = dict(available=False, error=str(error))
    models = {}
    for name in ('sam2.1-hiera-tiny-source', 'stable-diffusion-v1-5', 'instruct-pix2pix-editor', 'dinov2-small'):
        root = MODEL_ROOT / name
        weights = [p for p in root.rglob('*') if p.is_file() and p.suffix in ('.bin', '.safetensors')]
        models[name] = dict(directory_exists=root.is_dir(), weight_files=len(weights), ready=bool(weights))
    alex = Path(torch.hub.get_dir()) / 'checkpoints/alexnet-owt-7be5be79.pth'
    result = dict(project=str(PROJECT), requested_device=device, cuda_available=torch.cuda.is_available(),
                  packages=packages, local_models=models, perceptual_backbone_ready=alex.is_file(),
                  shared_pool_configured=SHARED_MEMORY.is_file(),
                  query_image_generation_implemented=False, external_model_calls=0)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result
