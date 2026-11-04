"""四阶段的文件摘要记录工具。"""
from pathlib import Path
from src.vision.image_utils import sha256_file
STAGES = ('foreground_segmentation', 'foreground_statistics', 'color_harmonization', 'latent_generation_refinement')
def stage_record(files, **facts):
    return dict(status='completed', files={str(Path(p).resolve()): sha256_file(p) for p in files}, **facts)
