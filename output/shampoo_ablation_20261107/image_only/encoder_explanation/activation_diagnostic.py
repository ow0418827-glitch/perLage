"""仅校正旧编码器激活函数的离线诊断，不修改任何源代码或正式结果。"""
import os
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import gc
from collections import Counter
import numpy as np
import torch
import open_clip
from src.paths import PROJECT
from src.visual.audit import read_json,freeze_json
from src.visual.image_utils import sha256_file
from src.visual.encoders.clip import make_encoder
from src.visual.visual_retrieval import resolve_image,unit

torch.set_num_threads(4)
root=PROJECT/'output/shampoo_ablation_20261107/image_only'
out=root/'encoder_explanation'
plan=read_json(root/'PLAN.json');base_rows=read_json(root/'RESULTS.json')
bank=read_json(root/'banks/clean.json')
source=next(i for i,e in enumerate(bank) if e['Id']==plan['source_memory_id'])
encoder=make_encoder(surrogate=2,device='cuda')
original=encoder.surrogate.model
before=Counter(type(m).__name__ for m in original.modules())
corrected=open_clip.create_model('ViT-B-32',pretrained=None,force_quick_gelu=True)
corrected.load_state_dict(original.state_dict(),strict=True)
corrected.eval().float().requires_grad_(False).to('cuda')
a=original.state_dict();b=corrected.state_dict()
assert a.keys()==b.keys() and all(torch.equal(a[k],b[k]) for k in a)
after=Counter(type(m).__name__ for m in corrected.modules())
assert before['GELU']==24 and before['QuickGELU']==0
assert after['GELU']==0 and after['QuickGELU']==24
encoder.surrogate.model=corrected
del original,a,b;gc.collect();torch.cuda.empty_cache()
cache={}
def encode(path):
    key=sha256_file(path)
    if key not in cache:cache[key]=unit(encoder.encode(path))
    return cache[key].copy()
vectors=np.stack([encode(resolve_image(e,root/'banks/clean.json')) for e in bank])
queries={k:encode(v['image']) for k,v in plan['queries'].items()}
variants={k:encode(v['image']) for k,v in plan['variants'].items()}
rows=[]
for name,v in variants.items():
    current=vectors.copy();current[source]=v
    for query,q in queries.items():
        scores=[float(x@q) for x in current]
        ordered=sorted(range(len(scores)),key=lambda i:(-scores[i],i))
        old=next(x for x in base_rows if (x['model'],x['variant'],x['query'])==('old_B32',name,query))
        row=dict(variant=name,query=query,cosine=scores[source],rank=ordered.index(source)+1,
            original_GELU_cosine=old['cosine'],original_GELU_rank=old['rank'])
        rows.append(row)
        print(name,query,'GELU',old['cosine'],old['rank'],'QuickGELU',row['cosine'],row['rank'],flush=True)
freeze_json(out/'ACTIVATION_CORRECTED_DIAGNOSTIC.json',dict(status='completed_diagnostic_only',
    changed_factor='GELU_to_pretrained_QuickGELU',all_weight_tensors_bitwise_equal=True,
    original_activation_counts=dict(before),corrected_activation_counts=dict(after),
    text_used=False,source_code_modified=False,formal_results_modified=False,
    existing_image_files_modified=False,bank_count=len(bank),rows=rows))
