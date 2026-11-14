"""离线诊断已停止的主色融合，不重训、不计算最终方法相似度、不覆盖失败证据。"""
import os
for name in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:
    os.environ[name]='4'
import sys
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[2]
sys.path.insert(0,str(PROJECT));sys.path.insert(0,str(ROOT/'implementation'))
import numpy as np
from PIL import Image
import torch
from src.vision._natural.colour import prepare_print_context,harmonize_statistics,HarmonizationConfig,halo_error
from src.vision._natural.core import source_ssim
import prototype_v2 as material


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def rgb(p):return np.asarray(Image.open(p).convert('RGB')).copy()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(4)
    gen=ROOT/'runs/dominant/07_shampoo/generation'
    definition=read(gen.parent/'definition.json');data=definition['data']
    initial=read(gen/'initialization/ready.json')
    output=ROOT/'diagnosis';output.mkdir(exist_ok=True)
    evidence=[gen/'color_harmonization/rejection.json',gen/'initialization/ready.json',
              gen/'initialization/attack/metadata.json',ROOT/'RUN_STATUS.json']
    hashes={str(p):sha(p) for p in evidence}
    source=rgb(data['source']);foreground=rgb(gen/'foreground_segmentation/source/foreground.png')[...,0]>0
    target=rgb(data['target']);target_fg=rgb(gen/'foreground_segmentation/target/foreground.png')[...,0]>0
    context=material.prepare(source,foreground,target,target_fg,definition['print_regions'])
    anchor=rgb(initial['candidates']['generation_only']['image'])
    old_final=rgb(initial['candidates']['natural_final']['image'])
    mask=np.asarray(Image.open(initial['mask_path']).convert('L')).copy();mask[context['protected']]=0
    legacy=prepare_print_context(source,anchor,mask,foreground,definition['print_regions'])
    if legacy['halo_applicable']:mask[legacy['opaque']]=0
    residual=old_final.astype(np.int16)-anchor.astype(np.int16);residual[mask==0]=0
    context['input_anchor']=anchor
    context['input_anchor_lab']=material.rgb2lab(anchor/255.)
    context['anchor_base_lab']=context['input_anchor_lab'][context['material']].mean(0)
    parent=halo_error(legacy,old_final);limit=parent*.5+1e-8
    halo=legacy['halo'];padding=halo&context['protected']
    raw_material=foreground&(np.linalg.norm(context['source_lab'][...,1:]-context['source_base'][1:],axis=-1)<=14.)
    raw_material &= context['source_lab'][...,0]>=context['source_base'][0]-28.
    raw_ink=legacy['roi']&foreground&~raw_material
    printed_core=context['protected']&~halo
    original=torch.from_numpy(source.copy()).float().permute(2,0,1)[None]/255
    trials=[]
    for strength in [0.,1/32,.25,.5,.78125,1.]:
        colored=material.render(context,strength,lightness_mix=.5,chroma_scale=1.)
        ring=prepare_print_context(source,colored,mask,foreground,definition['print_regions'])
        recomposed=harmonize_statistics(ring,mask,context['target_palette'][0]['rgb'],HarmonizationConfig(strength=.75))
        for name,restore in [('current_restore_all_protected',context['protected']),
                             ('diagnostic_keep_white_padding_recomposition',printed_core),
                             ('diagnostic_consistent_padding_transfer',printed_core)]:
            edited=recomposed.copy();edited[restore]=source[restore];edited[~foreground]=anchor[~foreground]
            final=np.clip(edited.astype(np.int16)+residual,0,255).astype(np.uint8)
            if name=='diagnostic_consistent_padding_transfer':
                ny,nx=legacy['nearest']
                local_delta=final[ny[halo],nx[halo]].astype(int)-source[ny[halo],nx[halo]].astype(int)
                final[halo]=np.clip(source[halo].astype(int)+local_delta,0,255).astype(np.uint8)
            x=torch.from_numpy(final.copy()).float().permute(2,0,1)[None]/255
            with torch.no_grad():
                ss=float(source_ssim(x,original));mae=float((x-original).abs().mean()*255)
            measured=halo_error(legacy,final)
            core_delta=int(np.abs(final.astype(int)-source.astype(int))[printed_core].max()) if printed_core.any() else 0
            row=dict(strength=strength,processing=name,source_ssim=ss,source_mae=mae,
                     halo_error=measured,halo_limit=limit,printed_core_max_difference=core_delta,
                     passes_fixed_limits=ss>=.85001 and mae<=12 and measured<=limit and core_delta==0)
            trials.append(row)
            if strength in [.5,.78125]:
                Image.fromarray(final).save(output/f'{name}_{strength:.5f}.png')
    overlay=source.copy();overlay[halo]=np.array([255,0,255]);Image.fromarray(overlay).save(output/'white_padding_halo_points.png')
    result=dict(scope='offline_failure_diagnosis_only_not_full_method_results',
        original_frozen_evidence_unchanged=all(sha(Path(p))==s for p,s in hashes.items()),
        evidence_sha256=hashes,halo_pixels=int(halo.sum()),
        halo_pixels_restored_by_final_protection=int(padding.sum()),
        halo_pixels_overlapping_raw_ink=int((halo&raw_ink).sum()),
        printed_core_pixels_without_white_padding=int(printed_core.sum()),
        parent_halo=parent,required_halo_maximum=limit,trials=trials,
        conclusion_zh='新融合实现先补偿印刷边缘底色，随后又把扩张保护范围内的白色边缘恢复原像素，抵消了补偿。该位置在既有光晕采样点中，导致保护与光晕减半要求冲突。',
        final_latent_refinement_started=False,full_new_image_available=False,
        final_similarity_computed=False,new_training_runs=0,main_program_modified=False)
    assert result['original_frozen_evidence_unchanged']
    (output/'FAILURE_DIAGNOSIS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['halo_pixels','halo_pixels_restored_by_final_protection','halo_pixels_overlapping_raw_ink','required_halo_maximum','trials']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
