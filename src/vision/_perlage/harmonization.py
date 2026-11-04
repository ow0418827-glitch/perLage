"""主体统计颜色融合；完整方法的内部第三阶段。"""
import numpy as np
import torch
from PIL import Image
from src.paths import relocate_io_path
from src.vision.image_utils import read_tensor, write_json
from src.vision._natural.colour import read_rgb, prepare_print_context, harmonize_statistics, HarmonizationConfig, halo_error

def _fuse(data,initial,foreground,statistics,regions,root,config,base):
    from src.vision._natural.core import source_ssim
    from src.vision.retrieval_objective import JointRetrievalObjective
    from src.vision.robust_objectives import transformed_batch
    output=root/'color_harmonization';output.mkdir(parents=True,exist_ok=True)
    source=read_rgb(data['source']);old_anchor=read_rgb(initial['candidates']['generation_only']['image'])
    old_final=read_rgb(initial['candidates']['natural_final']['image'])
    mask=np.asarray(Image.open(relocate_io_path(initial['mask_path'])).convert('L')).copy()
    context=prepare_print_context(source,old_anchor,mask,foreground,regions)
    mask[context['opaque']]=0
    residual=old_final.astype(np.int16)-old_anchor.astype(np.int16)
    if np.abs(residual).max()>4:raise ValueError('内部起点的辅助残差不合规')
    residual[mask==0]=0
    proposed=harmonize_statistics(context,mask,statistics['rgb'],HarmonizationConfig(**config.harmonization))
    device=base.surrogate.device
    original=read_tensor(data['source'],device)
    with torch.no_grad():
        target=base.surrogate.encode_image(read_tensor(data['target'],device)).global_embedding.detach()
        joint=JointRetrievalObjective(base.surrogate,target,data['source_title'],data['training_queries'])
        teacher=read_tensor(initial['candidates']['natural_final']['image'],device)
        floors=joint.scores(base.surrogate.encode_image(transformed_batch(teacher)).global_embedding)-config.latent['ordinary_tolerance']
        trials=[];chosen=None
        for index in range(32,-1,-1):
            fraction=index/32
            anchor=np.rint(old_anchor.astype(float)+fraction*(proposed.astype(float)-old_anchor)).astype(np.uint8)
            anchor[context['opaque']]=source[context['opaque']];anchor[~foreground]=old_anchor[~foreground]
            final=np.clip(anchor.astype(np.int16)+residual,0,255).astype(np.uint8)
            tensor=torch.as_tensor(final.copy(),device=device,dtype=torch.float32).permute(2,0,1)[None]/255
            quality=float(source_ssim(tensor,original));mae=float((tensor-original).abs().mean()*255)
            ordinary=joint.scores(base.surrogate.encode_image(transformed_batch(tensor)).global_embedding)
            guard=float((ordinary-floors).min())
            halo=halo_error(context,final);parent_halo=halo_error(context,old_final)
            accepted=(quality>=config.latent['source_ssim_floor']+1e-5 and mae<=config.latent['source_mae_ceiling']
                      and guard>=-1e-6 and halo<=parent_halo*.5+1e-8)
            trials.append(dict(fraction=fraction,source_ssim=quality,source_mae_levels=mae,
                               ordinary_guard_slack=guard,halo_error=halo,accepted=accepted))
            if accepted:chosen=(anchor,final);break
    if chosen is None:
        write_json(output/'rejection.json',dict(trials=trials));raise ValueError('统计颜色融合未通过固定质量与普通条件门槛')
    Image.fromarray(chosen[0]).save(output/'generative_anchor.png')
    Image.fromarray(chosen[1]).save(output/'initial.png');Image.fromarray(mask).save(output/'edit_mask.png')
    record=dict(status='completed',color_scheme='statistics',selected_rgb=statistics['rgb'],config=config.harmonization,
        chosen=trials[-1],trials=trials,halo_applicable=context['halo_applicable'],
        source_ssim_floor=config.latent['source_ssim_floor'],source_mae_ceiling=config.latent['source_mae_ceiling'],
        foreground_limited=True,printed_core_preserved=True,auxiliary_residual_fixed_for_latent_stage=True,
        private_gallery_used=False,formal_queries_used=False)
    write_json(output/'metadata.json',record)
    return output,record
