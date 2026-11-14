"""仅在全部完整图冻结之后计算公开图像和图文相似度，并生成报告。"""
import os
for name in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:
    os.environ[name]='4'
os.environ['TOKENIZERS_PARALLELISM']='false'
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[2]
sys.path.insert(0,str(PROJECT))
import numpy as np
import torch
import torch.nn.functional as F
from src.vision.image_utils import sha256_file
from src.vision.dual_rag_bridge import make_encoder
from src.vision.joint_encoder import ChunkedClipEncoder
from src.vision._perlage.evidence import validate_complete


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');temp.replace(p)


def freeze_all(plan):
    images={}
    for arm,group in plan['execution_order']:
        folder=ROOT/'runs'/arm/group
        completed=read(folder/'FULL_METHOD_COMPLETED.json')
        if completed.get('status')!='completed' or completed.get('final_latent_steps')!=60:
            raise ValueError('完整方法未完成：'+str(folder))
        ready=read(folder/'generation/ready.json')
        validate_complete(ready,completed['image'])
        frozen=folder/'frozen.png'
        if not frozen.exists():shutil.copyfile(completed['image'],frozen)
        if sha256_file(frozen)!=completed['image_sha256']:raise ValueError('冻结图摘要改变')
        images[(group,arm)]=str(frozen)
    for case in plan['cases']:
        if case['key']=='09_toothbrush':
            control=plan['control_brush'];ready=read(control['ready']);definition=read(control['definition'])
            if sha256_file(control['image'])!=control['sha256']:raise ValueError('牙刷完整对照改变')
            validate_complete(ready,control['image'])
            if ready['source_sha256']!=case['inputs_sha256']['source'] or ready['target_sha256']!=case['inputs_sha256']['target']:
                raise ValueError('牙刷完整对照图对身份不符')
            if definition['data']['training_queries']!=read(case['definition'])['data']['training_queries']:
                raise ValueError('牙刷对照训练指令不同')
            images[(case['key'],'original')]=control['image']
        definition=read(case['definition'])
        images[(case['key'],'source')]=definition['data']['source']
    payload={'status':'all_full_images_frozen','frozen_at':datetime.now(timezone.utc).isoformat(),
             'images':[{'group':k[0],'arm':k[1],'path':v,'sha256':sha256_file(v)} for k,v in images.items()],
             'no_color_step_early_stopping_used':True}
    path=ROOT/'ALL_FULL_IMAGES_FROZEN.json'
    if path.exists():
        old=read(path)
        if old['images']!=payload['images']:raise ValueError('完整图冻结清单改变')
    else:save(path,payload)
    return images


def calculate(plan, images):
    rows=[]
    for surrogate in [2,1]:
        base=make_encoder(surrogate=surrogate,device='cuda')
        enc=ChunkedClipEncoder(base)
        for case in plan['cases']:
            key=case['key'];definition=read(case['definition']);data=definition['data']
            target=torch.tensor(base.encode(data['target']),device=base.device)[None]
            title=torch.tensor(enc.encode_text(data['source_title']),device=base.device)[None]
            queries=torch.cat([F.normalize(target+torch.tensor(enc.encode_text(q),device=base.device)[None],dim=-1) for q in data['training_queries']])
            for arm in ['source','original','dominant']:
                feature=torch.tensor(base.encode(images[(key,arm)]),device=base.device)[None]
                row=dict(group=key,arm=arm,encoder=surrogate,
                         image_cosine=float((feature@target.T).item()),
                         joint_mean=float((F.normalize(feature+title,dim=-1)@queries.T).mean()),
                         image_sha256=sha256_file(images[(key,arm)]),model=base.snapshot)
                rows.append(row)
        del enc,base
        import gc;gc.collect();torch.cuda.empty_cache()
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');args=p.parse_args()
    torch.set_num_threads(4)
    plan=read(ROOT/'PLAN.json');images=freeze_all(plan)
    rows=calculate(plan,images)
    if args.verify:
        previous=read(ROOT/'FINAL_COMPARISON.json')['rows'];d=[]
        for a,b in zip(rows,previous):
            if [a[k] for k in ['group','arm','encoder','image_sha256']]!=[b[k] for k in ['group','arm','encoder','image_sha256']]:raise ValueError('复核身份不一致')
            d.extend(abs(a[k]-b[k]) for k in ['image_cosine','joint_mean'])
        maximum=max(d)
        if maximum>1e-6:raise ValueError('相似度独立复核不一致')
        save(ROOT/'INDEPENDENT_VERIFICATION.json',dict(passed=True,records=len(rows),scores=len(d),maximum_absolute_error=maximum,new_encoder_process=True))
        report(plan,images,previous)
        return
    comparisons=[]
    for key in ['07_shampoo','09_toothbrush']:
        for encoder in [2,1]:
            original=next(r for r in rows if (r['group'],r['encoder'],r['arm'])==(key,encoder,'original'))
            trial=next(r for r in rows if (r['group'],r['encoder'],r['arm'])==(key,encoder,'dominant'))
            comparisons.append(dict(group=key,encoder=encoder,image_gain=trial['image_cosine']-original['image_cosine'],joint_gain=trial['joint_mean']-original['joint_mean']))
    primary=[r for r in comparisons if r['encoder']==2];secondary=[r for r in comparisons if r['encoder']==1]
    gate=plan['promotion_reference']
    checks=dict(each_image=all(r['image_gain']>=gate['each_image_gain'] for r in primary),
                mean_image=float(np.mean([r['image_gain'] for r in primary]))>=gate['mean_image_gain'],
                each_joint=all(r['joint_gain']>=gate['each_joint_gain'] for r in primary),
                secondary_mean=float(np.mean([r['image_gain'] for r in secondary]))>=0)
    save(ROOT/'FINAL_COMPARISON.json',dict(status='completed_full_image_comparison',rows=rows,comparisons=comparisons,
         numerical_checks=checks,numerical_gate_passed=all(checks.values()),
         decision='candidate_requires_final_visual_review_before_promotion' if all(checks.values()) else 'retain_original_method',
         main_program_modified=False,shopping_runs=0,private_memory_used=False,formal_queries_used=False))


def report(plan,images,rows):
    result=read(ROOT/'FINAL_COMPARISON.json');check=read(ROOT/'INDEPENDENT_VERIFICATION.json')
    lines=['公子，本轮使用目标前景主色重新执行完整生成，最终相似度在全部图片冻结后计算。','',
           '洗发水与牙刷的新主色方案均从各自公开原图开始；洗发水原方法对照也重新完整生成。牙刷原方法对照复用经完整身份核验的既有最终图，训练进程不能读取它。','',
           '每张新生成图完整执行主体分割、前景统计选色、三十步指令生成、四十加四十加二十加六十步初始化优化、颜色融合及最终六十步潜变量细化。没有以颜色中间图相似度作停止判断。','',
           '|组别|编码器|未修改原图|原方法完整图|主色方案完整图|图像相似度增量|联合分数增量|','|---|---|---:|---:|---:|---:|---:|']
    for key,label in [('07_shampoo','洗发水'),('09_toothbrush','牙刷')]:
        for enc in [2,1]:
            rr={r['arm']:r for r in rows if r['group']==key and r['encoder']==enc}
            gain=next(x for x in result['comparisons'] if x['group']==key and x['encoder']==enc)
            lines.append(f"|{label}|{'主编码器' if enc==2 else '复核编码器'}|{rr['source']['image_cosine']:.6f}|{rr['original']['image_cosine']:.6f}|{rr['dominant']['image_cosine']:.6f}|{gain['image_gain']:+.6f}|{gain['joint_gain']:+.6f}|")
    lines += ['',f"独立复算：{check['records']}条记录、{check['scores']}个分数，最大误差{check['maximum_absolute_error']:.9g}。",'',
              '本轮是固定公开图对的完整方法开发对照，没有读取私有记忆或正式查询，没有执行购物。旧项目、既有正式图片与主程序均保持原样。','',
              '数值判断：'+('达到预先登记的增益门槛，待核对最终图的文字、图案与自然度后再替换主程序。' if result['numerical_gate_passed'] else '未达到预先登记的增益门槛，继续使用原方法；本次完整试验方案及产物单独保留。')]
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
