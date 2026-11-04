"""仅基于公开商品的、两种方法共用的离线图对筛选。"""
from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import re
import unicodedata

import numpy as np
from PIL import Image

from src.vision.image_utils import sha256_file


@dataclass(frozen=True)
class SelectionConfig:
    # 初始启发值，须在独立公开开发集校准；禁止用正式私有排名调整。
    shape_min: float = 0.65
    semantic_min: float = 0.55
    semantic_max: float = 0.90
    semantic_center: float = 0.72
    semantic_sigma: float = 0.12
    duplicate_title_similarity: float = 0.95
    mismatch_gamma: float = 2.0
    cost_temperature: float = 0.25
    aspect_temperature: float = 1.0
    weights: tuple = (0.35, 0.30, 0.25, 0.10)

    def validate(self):
        numbers = list(asdict(self).values())[:-1] + list(self.weights)
        if any(isinstance(x, bool) or not math.isfinite(x) for x in numbers):
            raise ValueError('选取参数必须为有限数值')
        if (not 0 <= self.shape_min <= 1
                or not -1 <= self.semantic_min < self.semantic_center < self.semantic_max <= 1
                or not self.semantic_max <= self.duplicate_title_similarity <= 1
                or min(self.semantic_sigma, self.cost_temperature, self.aspect_temperature) <= 0
                or self.mismatch_gamma < 0 or len(self.weights) != 4
                or min(self.weights) < 0 or sum(self.weights) <= 0):
            raise ValueError('选取阈值、窗口或权重无效')


def unit_rows(value):
    a = np.asarray(value, dtype=np.float64)
    if a.ndim not in (1, 2) or not a.size or not np.isfinite(a).all():
        raise ValueError('选取特征必须为有限、非空的一维或二维数组')
    norm = np.linalg.norm(a, axis=-1, keepdims=True)
    if np.any(norm <= 1e-12):
        raise ValueError('选取特征不能是零向量')
    return a / norm


def title_key(title):
    return re.sub(r'[\W_]+', '', unicodedata.normalize('NFKC', title).casefold())


def symmetric_patch_similarity(a, b):
    """两个方向各自平均最近块余弦，支持不同前景块数。"""
    a, b = unit_rows(a), unit_rows(b)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError('两组前景块特征维度不符')
    # 分块计算，避免候选高分辨率图片导致一次性占用平方级内存。
    def directed(x, y):
        return float(np.concatenate([(part @ y.T).max(axis=1)
                                     for part in np.array_split(x, max(1, math.ceil(len(x) / 256)))]).mean())
    return (directed(a, b) + directed(b, a)) / 2


def score_pair(source, target, source_features, target_features, perceptual_cost,
               config=SelectionConfig()):
    """输出可审计分项；分数是代理筛选量，不能解释为攻击成功概率。"""
    config.validate()
    reasons = []
    if source['id'] == target['id']:
        reasons.append('same_product_id')
    if title_key(source['title']) == title_key(target['title']):
        reasons.append('same_normalized_title')
    if source['image_sha256'] == target['image_sha256']:
        reasons.append('same_image_bytes')
    if source.get('owned') is not True:
        reasons.append('source_not_attacker_owned')
    text_cos = float(np.clip(unit_rows(source_features['text']) @ unit_rows(target_features['text']), -1, 1))
    cls = float(unit_rows(source_features['cls']) @ unit_rows(target_features['cls']))
    patches = symmetric_patch_similarity(source_features['patches'], target_features['patches'])
    shape = float(np.clip((cls + patches) / 2, 0, 1))
    aspect = math.exp(-abs(math.log(source_features['aspect'] / target_features['aspect'])) / config.aspect_temperature)
    if not math.isfinite(perceptual_cost) or perceptual_cost < 0:
        raise ValueError('感知改色代价必须有限且非负')
    if text_cos >= config.duplicate_title_similarity:
        reasons.append('near_duplicate_title')
    if shape < config.shape_min:
        reasons.append('shape_below_threshold')
    if not config.semantic_min <= text_cos <= config.semantic_max:
        reasons.append('semantic_outside_window')
    public_image = unit_rows(target_features['clip_image'])
    baseline = float(public_image @ unit_rows(target_features['clip_text']))
    cross = float(public_image @ unit_rows(source_features['clip_text']))
    mismatch = max(0.0, baseline - cross)
    components = dict(
        shape=shape * aspect,
        semantic=math.exp(-0.5 * ((text_cos - config.semantic_center) / config.semantic_sigma) ** 2),
        cross_modal=math.exp(-config.mismatch_gamma * mismatch),
        edit_cost=math.exp(-perceptual_cost / config.cost_temperature))
    weights = np.asarray(config.weights) / sum(config.weights)
    log_score = float(sum(w * math.log(max(v, 1e-12)) for w, v in zip(weights, components.values())))
    return dict(source_id=source['id'], target_id=target['id'], eligible=not reasons,
                rejected_reasons=reasons, score=math.exp(log_score), log_score=log_score,
                components=components, raw=dict(shape=shape, cls=cls, symmetric_patches=patches,
                aspect_penalty=aspect, semantic_cosine=text_cos, cross_modal_gap=mismatch,
                pilot_recolor_lpips=perceptual_cost))


def choose_pair(target, candidates, backend, config=SelectionConfig()):
    """无候选合格时停止；不降低门槛，也不读取私有排名补选。"""
    config.validate()
    ids = [r['id'] for r in candidates]
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError('公开候选标识不能重复或为空')
    target_features = backend.features(target)
    rows = []
    for product in candidates:
        features = backend.features(product)
        rows.append(score_pair(product, target, features, target_features,
                               backend.recolor_cost(product, target), config))
    eligible = sorted((r for r in rows if r['eligible']), key=lambda r: (-r['score'], r['source_id']))
    return dict(status='selected' if eligible else 'no_eligible_pair',
                selected_source_id=eligible[0]['source_id'] if eligible else None,
                target_id=target['id'], candidates=rows, configuration=asdict(config),
                calibration='provisional_public_only_heuristics_not_success_probabilities',
                models=backend.snapshot, victim_memory_used=False,
                victim_queries_used=False, target_system_probed=False)


def validate_product(record, base, *, owned=False):
    allowed = {'id', 'title', 'image', 'owned', 'source_url', 'source_note', 'boxes',
               'text_regions', 'text_regions_reviewed', 'foreground_mask'}
    if not isinstance(record, dict) or set(record) - allowed:
        raise ValueError('公开商品记录含未知字段；禁止传入历史、私有类目或检索结果')
    product = dict(record)
    for field in ('id', 'title', 'image'):
        if not isinstance(product.get(field), str) or not product[field].strip():
            raise ValueError('公开商品缺少有效字段：' + field)
    from src.vision.retrieval_objective import validate_public_title
    product['title'] = validate_public_title(product['title'])
    if owned and product.get('owned') is not True:
        raise ValueError('原商品必须明确声明为攻击者可修改的商品')
    if not (product.get('source_url') or product.get('source_note')):
        raise ValueError('须记录公开来源或攻击者自有资产说明')
    if product.get('text_regions_reviewed') is not True:
        raise ValueError('须先基于公开原图核对文字及品牌区域；无文字时显式标记空区域')
    if not isinstance(product.get('text_regions'), list):
        raise ValueError('须提供公开原图文字区域列表')
    for rect in product['text_regions']:
        if len(rect) != 4 or not all(math.isfinite(x) for x in rect) or not (0 <= rect[0] < rect[2] <= 1 and 0 <= rect[1] < rect[3] <= 1):
            raise ValueError('文字区域使用零至一的矩形坐标')
    for field in ('image', 'foreground_mask'):
        if product.get(field):
            path = Path(product[field])
            path = (path if path.is_absolute() else Path(base) / path).resolve()
            if not path.is_file():
                raise FileNotFoundError('公开图片或掩码不存在：' + str(path))
            product[field] = str(path)
    product['image_sha256'] = sha256_file(product['image'])
    if product.get('foreground_mask'):
        product['foreground_mask_sha256'] = sha256_file(product['foreground_mask'])
    if not product.get('foreground_mask') and not product.get('boxes'):
        raise ValueError('须提供基于公开原图的主体框或前景掩码')
    return product


class LocalSelectionBackend:
    """固定的本地公开代理；不调用受害系统，不下载权重或降级模型。"""
    def __init__(self, settings, output, device='cuda'):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        from sentence_transformers import SentenceTransformer
        import lpips
        from src.vision.dual_rag_bridge import make_encoder
        from src.vision.joint_encoder import ChunkedClipEncoder
        from src.paths import MODEL_ROOT
        self.device, self.output = device, Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        dino = str(settings.get('dino_model', MODEL_ROOT / 'dinov2-small'))
        text = settings.get('text_model', 'sentence-transformers/all-MiniLM-L6-v2')
        alex = Path(torch.hub.get_dir()) / 'checkpoints/alexnet-owt-7be5be79.pth'
        if not alex.is_file():
            raise FileNotFoundError('缺少感知度量主干权重；请先运行模型准备工具：' + str(alex))
        self.processor = AutoImageProcessor.from_pretrained(dino, local_files_only=True)
        self.dino = AutoModel.from_pretrained(dino, local_files_only=True).eval().requires_grad_(False).to(device)
        text_root = Path(text)
        if not text_root.is_dir():
            from huggingface_hub import snapshot_download
            text_root = Path(snapshot_download(text, local_files_only=True,
                allow_patterns=['*.json', '*.txt', 'model.safetensors', '1_Pooling/*']))
        self.text = SentenceTransformer(str(text_root.resolve()), local_files_only=True, device=device)
        self.clip = ChunkedClipEncoder(make_encoder(surrogate=1, device=device))
        self.lpips = lpips.LPIPS(net='alex', verbose=False).eval().requires_grad_(False).to(device)
        from src.vision.audit import model_files
        self.snapshot = dict(dino=model_files(Path(dino)), text_model=str(text),
                             clip=self.clip.snapshot, lpips=dict(backbone_path=str(alex), backbone_sha256=sha256_file(alex),
                             implementation='lpips_alex_v0.1'), implementation_sha256=sha256_file(__file__))
        # 标题编码使用模型实际本地配置，所有权重纳入身份。
        if text_root.is_dir():
            self.snapshot['text_files'] = model_files(text_root)
        else:
            raise ValueError('文本代理未报告可核验的本地权重目录')
        self.cache = {}

    def _prepared(self, product):
        from src.vision._natural.segmentation import segment_owned_image
        from src.vision._natural.colour import print_region_mask
        from src.vision.audit import digest
        image = np.asarray(Image.open(product['image']).convert('RGB')).copy()
        tag = digest(dict(image=product['image_sha256'], boxes=product.get('boxes'),
                          regions=product['text_regions'], mask=sha256_file(product['foreground_mask'])
                          if product.get('foreground_mask') else None))[:20]
        root = self.output / tag
        if product.get('foreground_mask'):
            foreground = np.asarray(Image.open(product['foreground_mask']).convert('L')) > 127
        else:
            foreground, _ = segment_owned_image(product['image'], product['boxes'], root / 'segmentation')
        if foreground.shape != image.shape[:2] or not foreground.any():
            raise ValueError('主体前景掩码尺寸不符或为空')
        printable = print_region_mask(image.shape, product['text_regions'])
        valid = foreground & ~printable
        if valid.sum() < max(16, foreground.sum() * 0.05):
            raise ValueError('去文字后的有效主体太少，须核对公开标注')
        ys, xs = np.where(foreground)
        box = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
        masked = image.copy()
        masked[~valid] = 127
        cropped, mask = masked[box], valid[box]
        h, w = cropped.shape[:2]
        side = max(h, w)
        canvas = np.full((side, side, 3), 127, dtype=np.uint8)
        canvas_mask = np.zeros((side, side), dtype=np.uint8)
        dy, dx = (side - h) // 2, (side - w) // 2
        canvas[dy:dy+h, dx:dx+w] = cropped
        canvas_mask[dy:dy+h, dx:dx+w] = mask * 255
        root.mkdir(parents=True, exist_ok=True)
        view = root / 'shape_view.png'
        Image.fromarray(canvas).resize((224, 224)).save(view)
        return dict(image=image, foreground=foreground, valid=valid, aspect=w/h,
                    shape_view=view, patch_mask=Image.fromarray(canvas_mask).resize((224, 224), Image.Resampling.NEAREST))

    def features(self, product):
        import torch
        import torch.nn.functional as F
        from src.vision.audit import digest
        key = digest(product)
        if key in self.cache:
            return self.cache[key]['features']
        prepared = self._prepared(product)
        image = Image.open(prepared['shape_view']).convert('RGB')
        inputs = self.processor(images=image, do_resize=False, do_center_crop=False, return_tensors='pt').to(self.device)
        with torch.no_grad():
            hidden = self.dino(**inputs).last_hidden_state[0].float()
        patch_size = int(self.dino.config.patch_size)
        grid = 224 // patch_size
        offset = 1 + int(getattr(self.dino.config, 'num_register_tokens', 0))
        patches = hidden[offset:]
        if patches.shape[0] != grid * grid:
            raise ValueError('视觉代理的前景块网格与预处理不符')
        mask = torch.from_numpy(np.asarray(prepared['patch_mask']).copy()).float()[None, None] / 255
        valid = F.adaptive_avg_pool2d(mask, (grid, grid)).flatten() >= 0.5
        if valid.sum() < 2:
            raise ValueError('视觉代理没有足够的非文字前景块')
        features = dict(cls=hidden[0].cpu().numpy(), patches=patches[valid.to(patches.device)].cpu().numpy(),
                        text=self.text.encode(product['title'], normalize_embeddings=True),
                        clip_image=self.clip.encode(prepared['shape_view']),
                        clip_text=self.clip.encode_text(product['title']), aspect=prepared['aspect'])
        self.cache[key] = dict(prepared=prepared, features=features)
        return features

    def recolor_cost(self, source, target):
        """衡量源图到固定试改色图的代价；不把两件不同商品的原始距离当编辑成本。"""
        import torch
        from skimage.color import rgb2lab, lab2rgb
        from src.vision.audit import digest
        self.features(source)
        self.features(target)
        a, b = (self.cache[digest(p)]['prepared'] for p in (source, target))
        lab = rgb2lab(a['image'] / 255.)
        target_lab = rgb2lab(b['image'] / 255.)
        delta = np.median(target_lab[b['valid']], axis=0) - np.median(lab[a['valid']], axis=0)
        delta[0] = 0  # 固定探针只改变色调，保留源图明暗、结构与文字。
        pilot = lab.copy()
        pilot[a['valid']] += 0.25 * delta
        changed = np.rint(lab2rgb(pilot).clip(0, 1) * 255).astype(np.uint8)
        changed[~a['valid']] = a['image'][~a['valid']]
        def tensor(array):
            im = Image.fromarray(array).resize((256, 256), Image.Resampling.BICUBIC)
            return torch.as_tensor(np.asarray(im).copy(), device=self.device).permute(2, 0, 1)[None].float() / 127.5 - 1
        with torch.no_grad():
            return max(0., float(self.lpips(tensor(a['image']), tensor(changed))))

    def close(self):
        self.clip.close()
        self.dino = self.text = self.lpips = None
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
