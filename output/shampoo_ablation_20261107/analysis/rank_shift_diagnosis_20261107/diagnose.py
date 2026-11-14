"""已冻结图片的离线归因诊断；旧项目及正式结果保持只读。"""
import os
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
import gc
import statistics
import time
from itertools import product
import numpy as np
import torch
from src.paths import PROJECT, ProjectPath as Path
from src.visual.audit import read_json,freeze_json,save_json
from src.visual.image_utils import sha256_file
from src.visual.visual_retrieval import VisualMemoryRetriever,resolve_image
from src.visual.encoders.clip import make_encoder
from src.visual.encoders.joint import ChunkedClipEncoder
from src.experiments.shared_evaluation import make_eval_encoder
from src.experiments.shampoo_ablation_evaluation import MemoizedEncoder

torch.set_num_threads(4)
run=PROJECT/'output/shampoo_ablation_20261107'
output=run/'analysis/rank_shift_diagnosis_20261107'
old=PROJECT.parent/'lage/src/output/lage_runs'
plan=read_json(run/'evaluation/PLAN.json')
tasks=read_json(old/'catalog_shampoo_defenses_20261029/tasks.json')
old_results=read_json(old/'catalog_shampoo_defenses_20261029/results.json')
new_results=read_json(run/'evaluation/RANKINGS.json')['rows']
old_expected=next(r for r in old_results if r['bank']=='clean' and r['condition']['label']=='none')['queries']
new_expected=next(r for r in new_results if r['variant']=='clean')['queries']
banks={
    'old15':old/'catalog_shampoo_defenses_20261029/banks/none/clean.json',
    'old100':old/'catalog_shared_memory_20261103/shared_memory_pool.json',
    'new189':Path(plan['pool_path']),
}
images={'labeled_target':Path(tasks[0]['query_image']),'unlabeled_reference':Path(plan['query_image'])}
texts={'old_english':[x['instruction'] for x in tasks],'new_chinese':[x['instruction'] for x in plan['queries']]}
prepared={}
for key,path in banks.items():
    document=read_json(path)
    entries=document['entries'] if isinstance(document,dict) else document
    selected=[x for x in entries if x['FinalASIN']==plan['source_asin']]
    assert len(selected)==1
    assert selected[0]['PrimaryImage']['SHA256']==plan['source_image_sha256']
    for row in entries:resolve_image(row,path)
    prepared[key]=(entries,selected[0]['Id'])
freeze_json(output/'PLAN.json',dict(scope='posthoc_read_only_rank_shift_factorial_diagnosis',
    source_asin=plan['source_asin'],source_image_sha256=plan['source_image_sha256'],
    banks={k:dict(path=str(v),sha256=sha256_file(v)) for k,v in banks.items()},
    images={k:dict(path=str(v),sha256=sha256_file(v)) for k,v in images.items()},
    texts=texts,models=['old_B32','current_g14'],
    image_and_title_weight=0.5,query_image_and_text_weight=0.5,index_text='title',
    generation_runs=0,shopping_runs=0,formal_results_modified=False,
    note='The text factor changes wording and language together; not a language-only causal test.'))
all_rows=[]
reproduction=[]
for model,factory in [('old_B32',lambda:ChunkedClipEncoder(make_encoder(surrogate=2,device='cuda'))),
                      ('current_g14',lambda:make_eval_encoder('cuda'))]:
    encoder=MemoizedEncoder(factory())
    started=time.perf_counter()
    try:
        for bank,(entries,source_id) in prepared.items():
            ranker=VisualMemoryRetriever(encoder,banks[bank],cache_dir=None)
            for image,text in product(images,texts):
                observations=[]
                for index,instruction in enumerate(texts[text]):
                    ranking=ranker.rank(entries,instruction,images[image])
                    assert len(ranking)==len(entries)
                    source=next(x for x in ranking if x['Id']==source_id)
                    best_other=next(x for x in ranking if x['Id']!=source_id)
                    observations.append(dict(request=index+1,rank=source['rank'],score=source['score'],
                        strongest_competitor_score=best_other['score'],
                        margin=source['score']-best_other['score']))
                    if index==0:
                        lookup={x['Id']:x for x in entries}
                        first_ranking=[dict(rank=x['rank'],score=x['score'],
                            asin=lookup[x['Id']]['FinalASIN'],title=lookup[x['Id']]['PrimaryImage']['Title'],
                            image_sha256=x['image_sha256']) for x in ranking]
                values=[x['rank'] for x in observations]
                row=dict(model=model,bank=bank,query_image=image,query_text=text,
                    bank_count=len(entries),mean_rank=statistics.mean(values),minimum_rank=min(values),
                    maximum_rank=max(values),first_query_rank=values[0],top1=sum(x==1 for x in values),
                    mean_score=statistics.mean(x['score'] for x in observations),
                    mean_margin=statistics.mean(x['margin'] for x in observations),
                    encoder=encoder.snapshot,queries=observations,first_query_complete_ranking=first_ranking)
                all_rows.append(row)
                save_json(output/'RESULTS.partial.json',all_rows)
                print(f'{model} {bank} {image} {text}: first={values[0]}, mean={row["mean_rank"]:.2f}, range={min(values)}-{max(values)}',flush=True)
                expected=(old_expected if (model,bank,image,text)==('old_B32','old15','labeled_target','old_english')
                          else new_expected if (model,bank,image,text)==('current_g14','new189','unlabeled_reference','new_chinese') else None)
                if expected is not None:
                    assert [q['rank'] for q in observations]==[q.get('rank',q.get('source_rank')) for q in expected]
                    error=max(abs(q['score']-e.get('score',e.get('source_score'))) for q,e in zip(observations,expected))
                    assert error<2e-5,error
                    reproduction.append(dict(model=model,bank=bank,status='passed',scores_rechecked=20,max_absolute_error=error))
        save_json(output/(model+'_DONE.json'),dict(status='completed',seconds=time.perf_counter()-started))
    finally:
        encoder.close()
        gc.collect();torch.cuda.empty_cache()
assert len(all_rows)==24 and len(reproduction)==2
freeze_json(output/'RESULTS.json',all_rows)
freeze_json(output/'REPRODUCTION_CHECKS.json',reproduction)
print('全部二十四个交叉条件完成；新旧原始条件复现通过。',flush=True)
