"""模型目录定位：先使用原加载库的缓存规则，再遍历完整快照。"""
import json
import os
from src.paths import ProjectPath as Path


def local_model_candidates(model, roots):
    """在明确限定的目录中查找手动下载模型及标准缓存布局。"""
    repo_name = 'models--' + model.replace('/', '--')
    for root in roots:
        root = Path(root).expanduser()
        for folder in (root, root/'models', root/'weights', root/'checkpoints'):
            yield folder / model
            yield folder / model.split('/')[-1]
            for cache in (folder, folder/'hub', folder/'huggingface'/'hub', folder/'.cache'/'huggingface'/'hub'):
                repo = cache if cache.name == repo_name else cache/repo_name
                ref = repo/'refs'/'main'
                if ref.is_file():
                    yield repo/'snapshots'/ref.read_text(encoding='utf-8').strip()
                if (repo/'snapshots').is_dir():
                    yield from sorted((repo/'snapshots').iterdir(), reverse=True)


def incomplete_reason(directory):
    if not (directory / 'config.json').is_file():return '缺少 config.json'
    indexes=[directory / n for n in ('model.safetensors.index.json','pytorch_model.bin.index.json') if (directory / n).is_file()]
    if indexes:
        for index in indexes:
            try:
                shards=set(json.loads(index.read_text(encoding='utf-8'))['weight_map'].values())
            except (ValueError,KeyError,TypeError):continue
            if shards and all((directory / shard).is_file() for shard in shards):return None
        return '权重索引无效或缺少权重分片'
    if (directory / 'model.safetensors').is_file() or (directory / 'pytorch_model.bin').is_file():return None
    return '缺少完整权重或分片索引'


def resolve_memory_model(config, base):
    from .backbone import _huggingface_hub_roots
    model=config.get('model','Alibaba-NLP/gme-Qwen2-VL-2B-Instruct' if config.get('provider')=='gme' else 'openai/clip-vit-base-patch32')
    model={'CLIP':'openai/clip-vit-base-patch32','CLIP-B-32':'openai/clip-vit-base-patch32'}.get(model.upper(),model)
    base=Path(base)
    roots=[Path(p).expanduser() for p in config.get('search_roots',[])]
    roots=[p if p.is_absolute() else base/p for p in roots]+[base,base.parent,Path.cwd()]
    explicit=Path(model).expanduser()
    candidates=[explicit] if explicit.is_absolute() else [root / explicit for root in roots]
    # Transformers 自身的缓存定位包含其版本对应的默认路径和旧缓存变量。
    TRANSFORMERS_CACHE=None
    try:
        from transformers.utils.hub import cached_file, TRANSFORMERS_CACHE
        cached=cached_file(model,'config.json',cache_dir=config.get('cache_dir'),revision=config.get('revision'),local_files_only=True)
        if cached:candidates.append(Path(cached).parent)
    except (OSError,ValueError,ImportError):
        pass
    caches=_huggingface_hub_roots()
    if config.get('cache_dir'):
        custom=Path(config['cache_dir']).expanduser()
        if not custom.is_absolute():custom=base / custom
        caches[:0]=[custom,custom/'hub']
    if TRANSFORMERS_CACHE:caches.append(Path(TRANSFORMERS_CACHE))
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        caches.append(Path(HF_HUB_CACHE))
    except ImportError:pass
    caches.append(Path(os.environ.get('XDG_CACHE_HOME',str(Path.home()/'.cache')))/'huggingface'/'hub')
    repo_name='models--'+model.replace('/','--')
    for cache in caches:
        repo=cache if cache.name==repo_name else cache/repo_name
        ref=repo/'refs'/(config.get('revision') or 'main')
        if ref.is_file():candidates.append(repo/'snapshots'/ref.read_text(encoding='utf-8').strip())
        if (repo/'snapshots').is_dir():candidates.extend(sorted((repo/'snapshots').iterdir(),reverse=True))
    # 允许复用原项目下常见的手工下载目录，不扫描整个磁盘。
    candidates.extend(local_model_candidates(model, roots))
    seen=set();checked=[]
    for candidate in candidates:
        key=str(candidate.absolute())
        if key in seen:continue
        seen.add(key)
        reason=incomplete_reason(candidate)
        if reason is None:return candidate.resolve()
        checked.append(f'{candidate}: {reason if candidate.exists() else "目录不存在"}')
    raise FileNotFoundError('未找到可加载的完整模型：'+model+'\n已检查：\n'+'\n'.join(checked)+'\n若原模型位于其他目录，请用 --memory-model 指定已有的完整模型目录；本检查不会下载模型。')
