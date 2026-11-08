"""lucid（公开参数 legacy_pixel） 的完整商品标题监督。

lucid 是当前项目原始像素适配方法的讨论名称，保留既有公开参数与行为。
本模块不修改相邻项目或检索文字策略。
"""

import hashlib
import math
from src.paths import ProjectPath as Path
import time


def validate_title_options(target_text, text_weight, method="legacy_pixel"):
    """保留标题原文；拒绝空白标题、无效权重和不支持的优化方法。"""
    if method not in ("legacy_pixel", "lucid"):
        raise ValueError("商品标题文字对齐仅支持 lucid（legacy_pixel）方法")
    if not isinstance(target_text, str) or not target_text.strip():
        raise ValueError("目标商品标题必须为非空文字，不能只包含空白")
    if isinstance(text_weight, bool):
        raise ValueError("标题文字权重必须为有限正数")
    try:
        weight = float(text_weight)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("标题文字权重必须为有限正数") from exc
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError("标题文字权重必须为有限正数")


def _model_context_length(model):
    """以已加载的文字模型为准，避免仅信任分词器的默认容量。"""
    for owner in (model, getattr(model, "text", None)):
        context = getattr(owner, "context_length", None)
        if context is not None:
            if isinstance(context, bool) or int(context) != context or context < 2:
                raise ValueError("文字模型的上下文容量无效")
            return int(context)
        positions = getattr(owner, "positional_embedding", None)
        if positions is not None:
            return int(positions.shape[0])
    raise ValueError("无法确认文字模型的真实上下文容量，已停止标题编码")


def tokenize_complete_title(surrogate, target_text):
    """审计现有标准分词器的完整序列，并验证实际输入没有被缩短或改写。"""
    import torch

    if not isinstance(target_text, str) or not target_text.strip():
        raise ValueError("目标商品标题必须为非空文字，不能只包含空白")
    tokenizer = surrogate.tokenizer
    # lucid 的三个代理与 PAPO 的第二代理均使用该标准分词器。
    # 未知分词器必须显式增加审计规则。
    if not all(hasattr(tokenizer, name) for name in ("encode", "sot_token_id", "eot_token_id")):
        raise ValueError("当前分词器不支持完整标题审计，不能保证不截断")
    if getattr(tokenizer, "reduction_fn", None) is not None:
        raise ValueError("标题监督不允许分词器采用随机删词或缩减策略")
    context = _model_context_length(surrogate.model)
    content = list(tokenizer.encode(target_text))
    sequence = [int(tokenizer.sot_token_id), *content, int(tokenizer.eot_token_id)]
    if len(sequence) > context:
        raise ValueError(
            f"完整商品标题需要 {len(sequence)} 个词元（含起止词元），"
            f"超过文字模型的 {context} 个词元容量；请更换文字模型，不能静默截断标题")
    # 额外的结束符会改变 CLIP 的池化位置，无法代表完整商品标题。
    if int(tokenizer.sot_token_id) in content or int(tokenizer.eot_token_id) in content:
        raise ValueError("商品标题包含模型的保留起止标记，不能作为完整标题编码")
    tokens = tokenizer([target_text], context_length=context)
    if not torch.is_tensor(tokens) or tuple(tokens.shape) != (1, context):
        raise ValueError("标题分词结果形状与文字模型容量不一致")
    actual = tokens[0].detach().cpu().tolist()
    if actual[:len(sequence)] != sequence or any(actual[len(sequence):]):
        raise ValueError("标题实际分词结果与完整序列不一致，已拒绝静默缩减")
    audit = {
        "tokenizer_class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        "model_context_length": context,
        "tokenizer_default_context_length": getattr(tokenizer, "context_length", None),
        "content_token_count": len(content),
        "special_token_count": 2,
        "total_token_count": len(sequence),
        "input_sequence_length": context,
        "truncated": False,
        "complete_sequence_verified": True,
        "token_ids_sha256": hashlib.sha256(tokens.detach().cpu().numpy().tobytes()).hexdigest(),
    }
    return tokens.to(surrogate.device), audit


class TitleAlignmentSurrogate:
    """只替换当前代理实例的文字编码；图片梯度继续由原代理提供。"""

    def __init__(self, surrogate, target_text):
        self.surrogate = surrogate
        self.target_text = target_text
        self.tokens, self.token_audit = tokenize_complete_title(surrogate, target_text)
        self.text_feature = None
        surrogate.model.eval()
        surrogate.model.requires_grad_(False)

    def __getattr__(self, name):
        return getattr(self.surrogate, name)

    def encode_text(self, text):
        import torch
        import torch.nn.functional as functional

        if text != self.target_text:
            raise ValueError("当前文字适配器只能编码已经审计的完整目标商品标题")
        if self.text_feature is None:
            with torch.no_grad():
                feature = self.surrogate.model.encode_text(self.tokens)
                if feature.ndim != 2 or feature.shape[0] != 1 or not torch.isfinite(feature).all():
                    raise ValueError("目标商品标题的文字向量无效")
                if torch.any(feature.norm(dim=-1) <= 0):
                    raise ValueError("目标商品标题的文字向量为零")
                self.text_feature = functional.normalize(feature, dim=-1).detach()
        return self.text_feature


def run_title_aligned_attack(source, target, output_dir, config, *, target_text,
                             steps, run_attack, surrogates=None):
    """将本地适配器注入原引擎，保存标题审计和落盘图片的文字对齐指标。"""
    import gc
    import torch
    from src.visual._lucid.objective import (
        TargetCache, prepare_targets, evaluate_objective)
    from src.visual._lucid.surrogates import load_surrogates, resolve_device
    from src.visual.io import load_image, pil_to_tensor, save_json

    validate_title_options(target_text, config.attack.text_weight)
    config.validate()
    device = resolve_device(config.device, allow_fallback=config.runtime.allow_device_fallback)
    models = result = adapter = targets = decoded = target_tensor = cache = weights = None
    started = time.perf_counter()
    try:
        models = list(surrogates) if surrogates is not None else load_surrogates(
            config.surrogates, device=str(device), precision=config.attack_precision,
            allow_downloads=config.runtime.allow_downloads,
            allow_device_fallback=config.runtime.allow_device_fallback,
            require_true_local_tokens=config.runtime.require_true_local_tokens,
            hash_weights=config.runtime.hash_model_weights)
        if len(models) != 3:
            raise ValueError("原始像素方法需要恰好三个代理模型")
        index = config.attack.text_model_index
        adapter = TitleAlignmentSurrogate(models[index], target_text)
        models[index] = adapter
        loading_seconds = time.perf_counter() - started
        cache = TargetCache()
        result = run_attack(Path(source), Path(target), target_text, config,
            method="legacy_pixel", steps=steps, output_directory=Path(output_dir),
            surrogates=models, target_cache=cache)
        metadata = result.metadata
        output = Path(metadata["output_image"])
        if not output.is_file():
            raise FileNotFoundError(output)
        # 原引擎的 final_objective 在八位落盘前计算；这里复用缓存重评实际落盘图片。
        target_tensor = pil_to_tensor(load_image(target), device=device)
        targets = prepare_targets(models, target_tensor, target_text, config.attack,
                                  cache=cache, prepare_cluster_centers=False)
        decoded = pil_to_tensor(load_image(output), device=device)
        weights = decoded.new_tensor(metadata["final_model_weights"])
        saved = evaluate_objective(decoded, models, targets, weights, config.attack,
                                   use_cached_target_centers=False).detached_record()
        metadata["after_attack_saved"] = saved
        metadata["text_alignment"] = {
            "enabled": True,
            "source": "target_product_title",
            "text_sha256": hashlib.sha256(target_text.encode("utf-8")).hexdigest(),
            "weight": float(config.attack.text_weight),
            "model_index": index,
            "model_snapshot": adapter.snapshot(),
            "token_audit": adapter.token_audit,
            "text_encoder_frozen": True,
            "candidate_selection": "last_completed_step",
            "surrogate_loading_and_title_audit_seconds": loading_seconds,
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        metadata["surrogates"] = [model.snapshot() for model in models]
        save_json(Path(metadata["output_metadata"]), metadata)
        return output, metadata
    finally:
        del models, result, adapter, targets, decoded, target_tensor, cache, weights
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
