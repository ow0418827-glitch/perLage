"""原优化引擎的独立适配器：三个代理、局部最优传输、像素/频域千步。"""
from dataclasses import replace
from pathlib import Path
import json
from .io import digest
from .model_paths import local_model_candidates
from .optimization.common.config import load_config
from .optimization.attack_core.surrogates import resolve_hf_cached_weight, load_surrogates
from .optimization.attack_core.engine import run_attack
from .optimization.attack_core.objective import TargetCache
from .optimization.frequency_attack.dct import DCTBasisCache


def resolve_optimization(path, surrogate_indices=None):
    path=Path(path).resolve()
    cfg=load_config(path)
    raw=json.loads(path.read_text(encoding='utf-8'))
    roots=[Path(p).expanduser() for p in raw.get('model_search_roots', [])]
    roots=[p if p.is_absolute() else path.parent/p for p in roots]
    models=[]
    for index, spec in enumerate(cfg.surrogates):
        if surrogate_indices is not None and index not in surrogate_indices:
            continue
        weight=Path(spec.weight_path).expanduser() if spec.weight_path else resolve_hf_cached_weight(spec.hf_repo)
        if weight is None and spec.hf_repo:
            weight=next((directory/'open_clip_model.safetensors'
                         for directory in local_model_candidates(spec.hf_repo, roots)
                         if (directory/'open_clip_model.safetensors').is_file()), None)
        if weight is not None and not weight.is_absolute():
            weight=path.parent/weight
        if weight is None or not weight.is_file():
            raise FileNotFoundError(f"缺少代理权重：{spec.name}/{spec.pretrained}；设置 weight_path 或准备本地缓存。额外搜索目录：{roots}")
        models.append(replace(spec,weight_path=str(weight.resolve())))
    # 新攻击只以乙图片为目标，关闭旧任务中面向目标文字的损失。
    return replace(cfg,surrogates=tuple(models),attack=replace(cfg.attack,text_weight=0.0))


class OriginalAttack:
    def __init__(self,config_path,options):
        self.config=resolve_optimization(config_path)
        self.options=options
        self.models=load_surrogates(self.config.surrogates,device=self.config.device,
                                   precision=self.config.attack_precision,allow_downloads=False)
        self.targets=TargetCache();self.bases=DCTBasisCache()
        self.snapshot={"engine":"ported_original", "models":[m.snapshot() for m in self.models]}

    def generate(self,source,target,output):
        result=run_attack(source,target,None,self.config,
                          method=self.options.get("method","legacy_pixel"),
                          steps=self.options.get("steps",1000),output_directory=Path(output).parent,
                          surrogates=self.models,target_cache=self.targets,basis_cache=self.bases)
        return {"engine":"ported_original","method":self.options.get("method","legacy_pixel"),
                "steps":self.options.get("steps",1000),"image_sha256":digest(output),
                "encoder":self.snapshot,"original_metrics":result.metadata,
                "text_target_used":False}
