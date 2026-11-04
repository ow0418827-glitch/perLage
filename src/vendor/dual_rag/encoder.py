"""可微视觉编码；攻击和检索可使用独立权重。支持独立选择记忆检索模型。"""
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from .io import digest


def image_tensor(path, device="cpu"):
    with Image.open(path) as image:
        array = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32) / 255


class ClipEncoder:
    def __init__(self, config, base):
        import open_clip
        self.device = torch.device(config.get("device", "cuda"))
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("当前环境没有可用图形处理器；可显式改为 cpu。")
        checkpoint = Path(config["checkpoint"])
        if not checkpoint.is_absolute():
            checkpoint = base / checkpoint
        if not checkpoint.is_file():
            raise FileNotFoundError(f"请配置本地编码器权重，不自动下载：{checkpoint}")
        self.model = open_clip.create_model(config["architecture"], pretrained=None)
        self.tokenizer = open_clip.get_tokenizer(config["architecture"])
        if checkpoint.suffix == ".safetensors":
            from safetensors.torch import load_file
            state = load_file(str(checkpoint))
        else:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            state = state.get("state_dict", state)
        state = {k.removeprefix("module."): v for k, v in state.items()}
        self.model.load_state_dict(state, strict=True)
        print("已严格载入本地编码器权重；后续不使用随机初始化参数。", flush=True)
        self.model.to(self.device).float().eval().requires_grad_(False)
        size = self.model.visual.image_size
        self.size = size if isinstance(size, int) else size[0]
        self.mean = torch.tensor((.48145466, .4578275, .40821073), device=self.device)[None, :, None, None]
        self.std = torch.tensor((.26862954, .26130258, .27577711), device=self.device)[None, :, None, None]
        self.snapshot = {"architecture": config["architecture"], "checkpoint_sha256": digest(checkpoint),
                         "precision": "float32", "preprocessing": "short_edge_bicubic_center_crop",
                         "provider": "clip", "fusion": "normalized_image_text_sum"}

    def encode_tensor(self, image):
        h, w = image.shape[-2:]
        scale = self.size / min(h, w)
        h2, w2 = max(self.size, round(h * scale)), max(self.size, round(w * scale))
        image = F.interpolate(image, size=(h2, w2), mode="bicubic", align_corners=False, antialias=True)
        image = image[:, :, (h2-self.size)//2:(h2-self.size)//2+self.size,
                      (w2-self.size)//2:(w2-self.size)//2+self.size]
        return F.normalize(self.model.encode_image((image-self.mean)/self.std).float(), dim=-1)

    @torch.no_grad()
    def encode(self, path):
        return self.encode_tensor(image_tensor(path, self.device))[0].cpu().numpy()

    @torch.no_grad()
    def encode_text(self, text):
        tokens=self.tokenizer([text]).to(self.device)
        return F.normalize(self.model.encode_text(tokens).float(),dim=-1)[0].cpu().numpy()


    def encode_joint(self, text, path):
        vector = self.encode(path) + self.encode_text(text)
        return vector / np.linalg.norm(vector)


from .model_paths import resolve_memory_model


class _Images(list):
    transform = None


class GMEEncoder:
    def __init__(self, config, base):
        from transformers import AutoModel
        local = resolve_memory_model(config, base)
        self.device = torch.device(config.get("device", "cuda"))
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("当前环境没有可用图形处理器。")
        dtype = torch.float16 if config.get("precision", "fp16") == "fp16" and self.device.type == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(str(local), trust_remote_code=True,
            local_files_only=True, torch_dtype=dtype,
            device_map="auto" if self.device.type == "cuda" else None).eval().requires_grad_(False)
        self.snapshot = {"provider":"gme", "model_path":str(local), "precision":str(dtype),
            "fusion":"gme_fused_embeddings", "image_max_side":224,
            "files":{str(p.relative_to(local)):digest(p) for p in sorted(local.rglob("*")) if p.is_file() and p.suffix in {".json", ".py", ".safetensors", ".bin", ".txt", ".model"}}}

    def _image(self, path):
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((224, 224))
            return im.copy()

    def _loader(self, path):
        return torch.utils.data.DataLoader(_Images([self._image(path)]), batch_size=1, num_workers=0, collate_fn=list)

    def _vector(self, result):
        if torch.is_tensor(result):
            result = result.detach().float().cpu().numpy()
        vector = np.asarray(result[0], dtype=np.float32)
        if not np.isfinite(vector).all() or not np.linalg.norm(vector):
            raise ValueError("GME 返回无效向量。")
        return vector / np.linalg.norm(vector)

    @torch.no_grad()
    def encode(self, path):
        return self._vector(self.model.get_image_embeddings(self._loader(path)))

    @torch.no_grad()
    def encode_text(self, text):
        return self._vector(self.model.get_text_embeddings([text or " "]))

    @torch.no_grad()
    def encode_joint(self, text, path):
        return self._vector(self.model.get_fused_embeddings(texts=[text], images=self._loader(path)))



class TransformersClipEncoder:
    """沿用原项目 AutoModel/AutoProcessor 加载及 MemoryEncoder 融合流程。"""
    def __init__(self, config, base):
        from transformers import AutoModel, AutoProcessor
        local = resolve_memory_model(config, base)
        self.device = torch.device(config.get("device", "cuda"))
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("当前环境没有可用图形处理器。")
        dtype = torch.float16 if config.get("precision", "fp16") == "fp16" and self.device.type == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(str(local), torch_dtype=dtype, local_files_only=True).to(self.device).eval()
        self.model.requires_grad_(False)
        self.processor = AutoProcessor.from_pretrained(str(local), local_files_only=True, use_fast=False)
        self.snapshot = {"provider":"clip", "loader":"transformers", "model_path":str(local),
            "precision":str(dtype), "fusion":"normalized_mean_clip_ablation", "image_max_side":224,
            "files":{str(p.relative_to(local)):digest(p) for p in sorted(local.rglob("*")) if p.is_file() and p.suffix in {".json", ".py", ".safetensors", ".bin", ".txt", ".model"}}}

    @torch.no_grad()
    def encode(self, path):
        with Image.open(path) as image:
            image=image.convert("RGB")
            image.thumbnail((224,224))
            inputs=self.processor(images=image,return_tensors="pt")
        features=self.model.get_image_features(**{k:v.to(self.device) for k,v in inputs.items()})
        if not torch.is_tensor(features):
            features=features.image_embeds if hasattr(features,"image_embeds") else features[0]
        return F.normalize(features.float(),dim=-1)[0].cpu().numpy()

    @torch.no_grad()
    def encode_text(self, text):
        inputs=self.processor(text=[str(text or " ").strip() or " "],return_tensors="pt",padding=True,truncation=True)
        features=self.model.get_text_features(**{k:v.to(self.device) for k,v in inputs.items()})
        if not torch.is_tensor(features):
            features=features.text_embeds if hasattr(features,"text_embeds") else features[0]
        return F.normalize(features.float(),dim=-1)[0].cpu().numpy()

    def encode_joint(self, text, path):
        vectors=[self.encode(path)]
        if text.strip():vectors.append(self.encode_text(text))
        vector=np.mean(vectors,axis=0)
        return vector/np.linalg.norm(vector)


class MatchedClipEncoder:
    """复用攻击代理的权重、正方形预处理、精度和全局特征路径。"""
    def __init__(self, config, base):
        from .optimization.common.config import SurrogateConfig
        from .optimization.attack_core.surrogates import OpenCLIPSurrogate
        spec = SurrogateConfig(**config['surrogate'])
        self.surrogate = OpenCLIPSurrogate(
            spec, device=config.get('device', 'cuda'), precision='fp32', allow_downloads=False)
        self.device = self.surrogate.device
        self.snapshot = {
            'provider': 'clip', 'loader': 'attack_surrogate',
            'surrogate_number': config['surrogate_number'],
            'surrogate': self.surrogate.snapshot(),
            'fusion': 'normalized_image_text_sum',
            'text_policy': 'clip_context_truncate',
        }

    @torch.no_grad()
    def encode(self, path):
        features = self.surrogate.encode_image(image_tensor(path, self.device))
        return features.global_embedding[0].float().cpu().numpy()

    @torch.no_grad()
    def encode_text(self, text):
        # 历史长文本不能使用攻击模块针对短文字目标的严格长度审计。
        # OpenCLIP tokenizer 的上下文截断行为被显式记录；纯图片对照不使用这些文本向量检索。
        tokens = self.surrogate.tokenizer([str(text or ' ')]).to(self.device)
        features = self.surrogate.model.encode_text(tokens)
        return F.normalize(features.float(), dim=-1)[0].cpu().numpy()

    def encode_joint(self, text, path):
        vector = self.encode(path) + self.encode_text(text)
        return vector / np.linalg.norm(vector)


def make_encoder(config, base):
    provider = config.get("provider", "clip")
    if provider == 'clip' and config.get('loader') == 'attack_surrogate':
        return MatchedClipEncoder(config, base)
    if provider == "clip":return TransformersClipEncoder(config, base)
    if provider == "gme":return GMEEncoder(config, base)
    raise ValueError(f"不支持的记忆编码器：{provider}")
