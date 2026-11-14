"""只读复算此前完整洗发水图与本次完整图；不生成图片、不读取正式查询。"""
import os
for key in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:
    os.environ[key]='4'
os.environ['TOKENIZERS_PARALLELISM']='false'
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[2]
sys.path.insert(0,str(PROJECT))
import numpy as np
import torch
import torch.nn.functional as F
from src.vision.dual_rag_bridge import make_encoder
from src.vision.joint_encoder import ChunkedClipEncoder
from src.vision._perlage.evidence import validate_complete


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def verify_old(folder,expected_method,source_sha,target_sha):
    ready=read(folder/'ready.json')
    if ready.get('method')!=expected_method or ready.get('status')!='completed':raise ValueError('历史方法身份不符')
    if ready['source_sha256']!=source_sha or ready['target_sha256']!=target_sha:raise ValueError('历史图对与本轮输入不同')
    required={'foreground_segmentation','foreground_statistics','color_harmonization','latent_generation_refinement'}
    if set(ready['stages'])!=required:raise ValueError('历史图缺少完整四阶段')
    for stage in ready['stages'].values():
        if stage['status']!='completed':raise ValueError('历史阶段未完成')
        for p,expected in stage['files'].items():
            if sha(p)!=expected:raise ValueError('历史完整图证据改变')
    final=ready['candidates']['final']
    if sha(final['image'])!=final['sha256']:raise ValueError('历史最终图摘要不符')
    latent=ready['stages']['latent_generation_refinement']
    if latent['steps_completed']!=latent['steps_requested']:raise ValueError('历史最终细化不完整')
    metadata=read(folder/'latent_generation_refinement/metadata.json')
    best=metadata.get('best',{})
    quality={k:best[k] for k in ['step','source_ssim','source_mae_levels','halo_error','parent_halo_error','quality_ok','natural_halo_gate'] if k in best}
    return ready,quality


def prepare():
    out=ROOT/'historical_comparison';out.mkdir(exist_ok=True)
    current=ROOT/'runs/dominant/07_shampoo'
    definition=read(current/'definition.json');data=definition['data']
    source_sha,target_sha=sha(data['source']),sha(data['target'])
    legacy_root=PROJECT.parent/'lage/src/output/lage_runs'
    original=legacy_root/'catalog_shampoo_20261028/generated/07_shampoo'
    no_shadow=legacy_root/'catalog_shampoo_no_shadow_20261029/generated/07_shampoo'
    old,old_quality=verify_old(original,'nf_lage',source_sha,target_sha)
    no_pool,no_pool_quality=verify_old(no_shadow,'nf_lage_no_shadow',source_sha,target_sha)
    actual_reused=PROJECT/'src/output/shampoo_k13_20261106/inputs/07_shampoo/modified.png'
    if sha(actual_reused)!=old['candidates']['final']['sha256']:raise ValueError('旧实际测试使用图与冻结图不符')
    ready=read(current/'generation/ready.json');completed=read(current/'FULL_METHOD_COMPLETED.json')
    validate_complete(ready,current/'frozen.png')
    images={'source':Path(data['source']),
            'old_actual_tested_complete':actual_reused,
            'old_no_shadow_complete':Path(no_pool['candidates']['final']['image']),
            'new_temporary_relaxed_complete':current/'frozen.png'}
    frozen={'created_utc':datetime.now(timezone.utc).isoformat(),'scope':'existing_complete_images_same_metric_public_target_comparison',
            'source_sha256':source_sha,'target_sha256':target_sha,
            'images':{name:{'path':str(p),'sha256':sha(p)} for name,p in images.items()},
            'old_actual_method':old['method'],'old_no_shadow_method':no_pool['method'],
            'old_actual_quality':old_quality,'old_no_shadow_quality':no_pool_quality,
            'new_quality':{k:completed['quality'][k] for k in ['step','source_ssim','source_mae_levels','halo_error','parent_halo_error','quality_ok','natural_halo_gate']},
            'existing_old_report':str(original.parent.parent/'REPORT.md'),
            'new_generation_runs':0,'original_project_modified':False,'formal_queries_used':False}
    path=out/'IMAGE_BINDING.json'
    if path.exists():
        if read(path)['images']!=frozen['images']:raise ValueError('本次比较的图片身份改变')
    else:save(path,frozen)
    return out,data,images,frozen


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--verify',action='store_true');args=parser.parse_args()
    torch.set_num_threads(4)
    out,data,images,binding=prepare()
    rows=[]
    for number in [2,1]:
        base=make_encoder(surrogate=number,device='cuda');encoder=ChunkedClipEncoder(base)
        target=torch.tensor(base.encode(data['target']),device=base.device)[None]
        title=torch.tensor(encoder.encode_text(data['source_title']),device=base.device)[None]
        queries=torch.cat([F.normalize(target+torch.tensor(encoder.encode_text(q),device=base.device)[None],dim=-1) for q in data['training_queries']])
        for name,path in images.items():
            feature=torch.tensor(base.encode(path),device=base.device)[None]
            rows.append(dict(encoder=number,image=name,sha256=sha(path),
                             image_cosine=float((feature@target.T).item()),
                             joint_mean=float((F.normalize(feature+title,dim=-1)@queries.T).mean()),
                             model=base.snapshot))
        del base,encoder
        import gc;gc.collect();torch.cuda.empty_cache()
    if args.verify:
        old=read(out/'METRICS.json');errors=[]
        for a,b in zip(rows,old['rows']):
            if [a[k] for k in ['encoder','image','sha256']]!=[b[k] for k in ['encoder','image','sha256']]:raise ValueError('复算身份不符')
            errors.extend(abs(a[k]-b[k]) for k in ['image_cosine','joint_mean'])
        if len(rows)!=8 or max(errors)>1e-6:raise ValueError('历史比较独立复核失败')
        save(out/'INDEPENDENT_VERIFICATION.json',dict(passed=True,records=len(rows),scores=len(errors),maximum_absolute_error=max(errors),fresh_encoder_process=True))
        print('历史图比较独立复核通过，最大误差',max(errors));return
    previous=read(ROOT/'COMPLETED_NEW_SHAMPOO_METRICS.json')
    for number in [2,1]:
        a=next(r for r in rows if r['encoder']==number and r['image']=='new_temporary_relaxed_complete')
        b=next(r for r in previous['rows'] if r['encoder']==number and r['image']=='complete_new_final')
        if max(abs(a[k]-b[k]) for k in ['image_cosine','joint_mean'])>1e-6:raise ValueError('新图与上一条回答分数口径不一致')
    comparisons=[]
    for number in [2,1]:
        rr={r['image']:r for r in rows if r['encoder']==number};new=rr['new_temporary_relaxed_complete']
        for old in ['old_actual_tested_complete','old_no_shadow_complete']:
            comparisons.append(dict(encoder=number,reference=old,image_delta=new['image_cosine']-rr[old]['image_cosine'],
                                    joint_delta=new['joint_mean']-rr[old]['joint_mean']))
    save(out/'METRICS.json',dict(scope=binding['scope'],rows=rows,comparisons=comparisons,
        same_target_same_title_same_generic_queries_same_weights_same_preprocessing=True,
        newly_generated_images=0,historical_training_configs_differ=True,
        not_a_single_factor_color_ablation=True,private_memory_used=False,shopping_runs=0))
    print(json.dumps({'rows':[{k:r[k] for k in ['encoder','image','image_cosine','joint_mean']} for r in rows],
                      'comparisons':comparisons,'quality':{k:binding[k] for k in ['old_actual_quality','old_no_shadow_quality','new_quality']}},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
