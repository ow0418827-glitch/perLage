"""冻结并评估已完成的洗发水新最终图；不把未完成原方法对照冒充有效结果。"""
import os
for name in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:
    os.environ[name]='4'
os.environ['TOKENIZERS_PARALLELISM']='false'
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[2]
sys.path.insert(0,str(PROJECT));sys.path.insert(0,str(ROOT/'implementation'))
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.color import rgb2lab,lab2rgb
from src.vision._perlage.evidence import validate_complete
from src.vision.dual_rag_bridge import make_encoder
from src.vision.joint_encoder import ChunkedClipEncoder
import prototype_v2 as material_impl


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def rgb(path):return np.asarray(Image.open(path).convert('RGB')).copy()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--verify',action='store_true');args=parser.parse_args()
    torch.set_num_threads(4)
    folder=ROOT/'runs/dominant/07_shampoo';gen=folder/'generation'
    completed=read(folder/'FULL_METHOD_COMPLETED.json');ready=read(gen/'ready.json')
    validate_complete(ready,completed['image'])
    if completed['final_latent_steps']!=60 or not completed['all_four_stages_completed']:
        raise ValueError('新图没有完整生成')
    frozen=folder/'frozen.png'
    if not frozen.exists():shutil.copyfile(completed['image'],frozen)
    if sha(frozen)!=completed['image_sha256']:raise ValueError('最终图摘要改变')
    definition=read(folder/'definition.json');data=definition['data']
    initial=read(gen/'initialization/ready.json')
    images={'unmodified_source':data['source'],
            'complete_initialization_before_color':initial['candidates']['natural_final']['image'],
            'color_seed_before_final_refinement':str(gen/'color_harmonization/initial.png'),
            'complete_new_final':str(frozen)}
    binding={name:{'path':str(p),'sha256':sha(p)} for name,p in images.items()}
    freeze=ROOT/'COMPLETED_NEW_SHAMPOO_FROZEN.json'
    if freeze.exists():
        if read(freeze)['images']!=binding:raise ValueError('已完成图评估输入改变')
    else:save(freeze,dict(scope='completed_new_method_only_original_control_pending',frozen_at=datetime.now(timezone.utc).isoformat(),images=binding,target_sha256=sha(data['target']),all_new_training_finished=True))
    rows=[]
    for number in [2,1]:
        base=make_encoder(surrogate=number,device='cuda');enc=ChunkedClipEncoder(base)
        target=torch.tensor(base.encode(data['target']),device=base.device)[None]
        title=torch.tensor(enc.encode_text(data['source_title']),device=base.device)[None]
        queries=torch.cat([F.normalize(target+torch.tensor(enc.encode_text(q),device=base.device)[None],dim=-1) for q in data['training_queries']])
        for label,path in images.items():
            feature=torch.tensor(base.encode(path),device=base.device)[None]
            rows.append(dict(encoder=number,image=label,path=str(path),sha256=sha(path),
                             image_cosine=float((feature@target.T).item()),
                             joint_mean=float((F.normalize(feature+title,dim=-1)@queries.T).mean()),
                             model=base.snapshot))
        del enc,base
        import gc;gc.collect();torch.cuda.empty_cache()
    path=ROOT/'COMPLETED_NEW_SHAMPOO_METRICS.json'
    if args.verify:
        old=read(path);errors=[]
        for a,b in zip(rows,old['rows']):
            if [a[k] for k in ['encoder','image','sha256']]!=[b[k] for k in ['encoder','image','sha256']]:raise ValueError('复核身份不一致')
            errors.extend(abs(a[k]-b[k]) for k in ['image_cosine','joint_mean'])
        if len(rows)!=8 or max(errors)>1e-6:raise ValueError('已完成最终图分数复核失败')
        save(ROOT/'COMPLETED_NEW_SHAMPOO_VERIFICATION.json',dict(passed=True,records=len(rows),scores=len(errors),maximum_absolute_error=max(errors),fresh_encoder_process=True))
        print('已完成新图独立复核，最大误差',max(errors));return
    source=rgb(data['source']);fg=rgb(gen/'foreground_segmentation/source/foreground.png')[...,0]>0
    target_fg=rgb(gen/'foreground_segmentation/target/foreground.png')[...,0]>0
    context=material_impl.prepare(source,fg,rgb(data['target']),target_fg,definition['print_regions'])
    mask=context['material'];target_lab=np.array(context['target_palette'][0]['lab'])
    colors=[]
    for name,p in images.items():
        lab=rgb2lab(rgb(p)/255.);mean=lab[mask].mean(0)
        colors.append(dict(image=name,material_mean_rgb=np.rint(lab2rgb(mean[None,None])[0,0]*255).astype(int).tolist(),
                           material_mean_lab=mean.tolist(),target_color_distance=float(np.linalg.norm(mean-target_lab)),
                           target_chroma_distance=float(np.linalg.norm(mean[1:]-target_lab[1:]))))
    result=dict(scope='completed_new_final_vs_unmodified_source_and_its_training_stages',
        full_new_generation_completed=True,original_current_method_control_completed=False,
        rows=rows,colors=colors,target_dominant_rgb=context['target_palette'][0]['rgb'],
        quality=completed['quality'],protected_core_max_difference=completed['protected_core_max_difference'],
        default_halo_limit=max(completed['quality']['parent_halo_error']*.5,.5),
        temporary_halo_limit=max(completed['quality']['parent_halo_error']*2.,.5),
        final_image_sha256=sha(frozen),main_program_modified=False,
        comparison_to_original_method_available=False,shopping_runs=0,private_memory_used=False)
    save(path,result)
    print(json.dumps({'rows':[{k:r[k] for k in ['encoder','image','image_cosine','joint_mean']} for r in rows],
                      'colors':colors},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
