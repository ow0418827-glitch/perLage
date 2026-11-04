"""对共享池所有图片施加相同防御，查询图片保持原样。"""
from pathlib import Path
import io
from PIL import Image, ImageFilter

from src.vision.audit import digest, read_json, save_json
from src.vision.image_utils import sha256_file


def conditions():
    result = [dict(label='none', kind='none', size=512)]
    result += [dict(label=f'blur{str(r).replace(".", "p")}', kind='blur', radius=r, size=512) for r in (0.5, 1, 2, 4)]
    result += [dict(label=f'jpeg{q}', kind='jpeg', quality=q, size=512) for q in (90, 75, 50, 25)]
    result += [dict(label='resize512', kind='resize', size=512), dict(label='vae512', kind='vae', size=512)]
    result += [dict(label=f'diff512_t{t}_s{s}', kind='diffusion', size=512, timestep=t, seed=s)
               for t in (99, 199, 399) for s in range(1001, 1006)]
    return result


def table_groups():
    rows = [('无防御', ['none'])]
    rows += [(f'模糊半径 {r}', [f'blur{str(r).replace(".", "p")}']) for r in (0.5, 1, 2, 4)]
    rows += [(f'压缩质量 {q}', [f'jpeg{q}']) for q in (90, 75, 50, 25)]
    rows += [('缩放重建', ['resize512']), ('变分自编码重建', ['vae512'])]
    rows += [(f'扩散净化起点 {t}', [f'diff512_t{t}_s{s}' for s in range(1001, 1006)]) for t in (99, 199, 399)]
    return rows


class Defender:
    def __init__(self, output, device='cuda'):
        self.output, self.device, self.purifier = Path(output), device, None

    def __call__(self, image_path, spec):
        if spec['kind'] == 'none':
            return Path(image_path).resolve()
        import PIL
        identity = dict(input_sha256=sha256_file(image_path), condition=spec,
                        implementation_sha256=sha256_file(__file__), pillow=PIL.__version__)
        if spec['kind'] in ('vae', 'diffusion'):
            if self.purifier is None:
                from src.vision.diffusion_purification import LatentDiffusionPurifier, PurificationConfig
                self.purifier = LatentDiffusionPurifier(PurificationConfig(working_size=512), device=self.device)
            identity['purifier'] = self.purifier.snapshot()
        path = self.output / spec['label'] / (digest(identity) + '.png')
        sidecar = path.with_suffix('.json')
        if path.exists() or sidecar.exists():
            if not path.is_file() or not sidecar.is_file():
                raise ValueError('防御缓存不完整')
            saved = read_json(sidecar)
            if saved['identity'] != identity or saved['output_sha256'] != sha256_file(path):
                raise ValueError('防御缓存图片或身份已改变')
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(image_path) as opened:
            image = opened.convert('RGB')
        if spec['kind'] == 'blur':
            image.filter(ImageFilter.GaussianBlur(spec['radius'])).save(path)
        elif spec['kind'] == 'jpeg':
            stream = io.BytesIO()
            image.save(stream, format='JPEG', quality=spec['quality'], subsampling=2)
            stream.seek(0)
            with Image.open(stream) as decoded:
                decoded.convert('RGB').save(path)
        elif spec['kind'] == 'resize':
            from src.vision.diffusion_purification import working_shape
            h, w = working_shape((image.height, image.width), spec['size'])
            image.resize((w, h), Image.Resampling.BICUBIC).resize(image.size, Image.Resampling.BICUBIC).save(path)
        elif spec['kind'] in ('vae', 'diffusion'):
            import torch
            from src.vision.image_utils import read_tensor, save_tensor_image
            tensor = read_tensor(image_path, self.purifier.device)
            with torch.no_grad():
                output = (self.purifier.reconstruct(tensor) if spec['kind'] == 'vae'
                          else self.purifier(tensor, seed=spec['seed'], start_timestep=spec['timestep']))
            save_tensor_image(path, output)
        else:
            raise ValueError('未知防御设置')
        save_json(sidecar, dict(identity=identity, output_sha256=sha256_file(path)))
        return path
