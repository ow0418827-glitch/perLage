"""同一全局库的纯图片检索；禁止任何文字特征调用。"""
import os
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
import copy
import gc
import numpy as np
import torch
from src.paths import PROJECT,ProjectPath as Path
from src.visual.audit import read_json,freeze_json,save_json
from src.visual.image_utils import sha256_file
from src.visual.visual_retrieval import VisualMemoryRetriever,resolve_image,unit
from src.visual.encoders.clip import make_encoder
from src.experiments.shared_evaluation import make_eval_encoder

torch.set_num_threads(4)
root=PROJECT/'output/shampoo_ablation_20261107'
output=root/'image_only'
previous=read_json(root/'evaluation/PLAN.json')
labels={'clean':'未修改源图','lucid_4':'像素方法，4/255','lucid_8':'像素方法，8/255',
    'lucid_original_16':'像素方法，旧16/255',
    'perlage_original_no_shadow':'历史完整感知图',
    'perlage_pixel_control':'本轮感知方法，含像素对照',
    'perlage_latent_only':'本轮感知方法，无像素版本'}
pool=read_json(previous['pool_path'])['entries']
assert len(pool)==189
bank=[]
for e in pool:
    image=resolve_image(e,previous['pool_path'])
    bank.append(dict(Id=e['Id'],FinalASIN=e['FinalASIN'],PrimaryImage=dict(
        ASIN=e['FinalASIN'],LocalPath=str(image),SHA256=sha256_file(image))))
source_index=next(i for i,e in enumerate(bank) if e['Id']==previous['source_memory_id'])
variants={name:previous['variants'][name] for name in labels}
queries={'target_original':dict(image=previous['target_image'],sha256=previous['target_image_sha256']),
         'unlabeled_reference':dict(image=previous['query_image'],sha256=previous['query_image_sha256'])}
assert all(sha256_file(x['image'])==x['sha256'] for x in [*variants.values(),*queries.values()])
freeze_json(output/'PLAN.json',dict(protocol='image_only_query_and_memory_v1',
    pool_path=previous['pool_path'],pool_sha256=previous['pool_sha256'],pool_count=189,
    source_memory_id=previous['source_memory_id'],source_asin=previous['source_asin'],
    queries=queries,variants=variants,labels=labels,models=['old_B32','current_g14'],
    image_weight=1.0,query_text_used=False,product_titles_used=False,purchase_history_used=False,
    key_mode='image',query_mode='image',only_source_image_replaced=True,
    primary_query='target_original',generation_runs=0,shopping_runs=0))

class ImageOnlyEncoder:
    supports_image_weight=True
    def __init__(self,base):
        self.base=base;self.snapshot=base.snapshot;self.device=base.device
        self.cache={};self.image_model_calls=0;self.text_calls=0
    def encode(self,path):
        key=sha256_file(path)
        if key not in self.cache:
            self.cache[key]=self.base.encode(path);self.image_model_calls+=1
        return self.cache[key].copy()
    def encode_text(self,*args,**kwargs):
        self.text_calls+=1;raise AssertionError('纯图片实验禁止文字编码')
    def encode_joint(self,*args,**kwargs):
        self.text_calls+=1;raise AssertionError('纯图片实验禁止图文融合')
    def close(self):
        if hasattr(self.base,'close'):self.base.close()
        self.cache.clear();self.base=None

results=[];audits=[]
for model,factory in [('old_B32',lambda:make_encoder(surrogate=2,device='cuda')),
                      ('current_g14',lambda:make_eval_encoder('cuda'))]:
    encoder=ImageOnlyEncoder(factory())
    try:
        original_vectors=np.stack([unit(encoder.encode(e['PrimaryImage']['LocalPath'])) for e in bank])
        query_vectors=np.stack([unit(encoder.encode(x['image'])) for x in queries.values()])
        variant_vectors=np.stack([unit(encoder.encode(x['image'])) for x in variants.values()])
        np.savez(output/(model+'_features.npz'),bank_vectors=original_vectors,
            query_vectors=query_vectors,variant_vectors=variant_vectors,source_index=source_index,
            entry_ids=np.array([e['Id'] for e in bank]),query_names=np.array(list(queries)),
            variant_names=np.array(list(variants)))
        for name,v in variants.items():
            current=copy.deepcopy(bank)
            if name!='clean':current[source_index]['RetrievalImage']=dict(ASIN=previous['source_asin'],LocalPath=v['image'],SHA256=v['sha256'])
            path=output/'banks'/(name+'.json');freeze_json(path,current)
            ranker=VisualMemoryRetriever(encoder,path,key_mode='image',query_mode='image',
                key_image_weight=1,query_image_weight=1,cache_dir=None)
            for query,q in queries.items():
                ranked=ranker.rank(current,'',q['image'])
                assert len(ranked)==189 and len({x['Id'] for x in ranked})==189
                source=next(e for e in ranked if e['Id']==previous['source_memory_id'])
                results.append(dict(model=model,variant=name,label=labels[name],query=query,
                    rank=source['rank'],cosine=source['score'],bank_count=189,
                    ranking=[dict(Id=x['Id'],rank=x['rank'],cosine=x['score'],image_sha256=x['image_sha256']) for x in ranked]))
                print(f'{model} {name} {query}: rank={source["rank"]}, cosine={source["score"]:.6f}',flush=True)
            save_json(output/'RESULTS.partial.json',results)
        assert encoder.text_calls==0
        audits.append(dict(model=model,encoder=encoder.snapshot,text_encoding_calls=encoder.text_calls,
            image_encoding_calls=encoder.image_model_calls,bank_titles_removed=True,
            query_instruction_empty=True,feature_file_sha256=sha256_file(output/(model+'_features.npz'))))
    finally:
        encoder.close();del encoder;gc.collect();torch.cuda.empty_cache()
assert len(results)==28
freeze_json(output/'RESULTS.json',results)
freeze_json(output/'ENCODING_AUDIT.json',audits)
old_sim={x['variant']:x for x in read_json(root/'evaluation/SIMILARITIES.json')}
errors=[]
for row in results:
    model='training_proxy_2' if row['model']=='old_B32' else 'held_out'
    field='target_image_cosine' if row['query']=='target_original' else 'query_image_cosine'
    error=abs(row['cosine']-old_sim[row['variant']][model][field]);errors.append(error)
    assert error<=2e-5
freeze_json(output/'EXISTING_IMAGE_COSINE_CHECK.json',dict(status='passed',comparisons=len(errors),max_absolute_error=max(errors)))
print('纯图片检索完成，文字编码调用为零。',flush=True)
