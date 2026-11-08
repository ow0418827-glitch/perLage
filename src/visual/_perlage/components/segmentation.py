from src.paths import model_path
"""主体分割实现单独冻结，不把后续颜色融合改动混入分割缓存身份。"""
from src.paths import relocate_io_path
from src.paths import ProjectPath as Path
from src.visual.image_utils import sha256_file,write_json
from src.visual._perlage.components.assets import read_json
import numpy as np
from PIL import Image
import gc,time,torch

def read_rgb(path):return np.asarray(Image.open(relocate_io_path(path)).convert('RGB')).copy()


def foreground_choices(masks,scores,boxes):
    choices=[];flags=[]
    for index,box in enumerate(boxes):
        x,y=round((box[0]+box[2])/2),round((box[1]+box[3])/2)
        valid=np.asarray(masks[index,:,y,x],dtype=bool)
        eligible=np.flatnonzero(valid)
        if not len(eligible):raise ValueError('所有主体候选均排除了自有商品框中心，需改进分割提示')
        choices.append(int(eligible[np.argmax(scores[index,eligible])]))
        flags.append(valid.tolist())
    return choices,flags


def segment_owned_image(path,boxes,output,model_path=model_path('sam2.1-hiera-tiny-source')):
    """框来自调用者的自有图；只按模型自身预测置信度选分割。"""
    from transformers import Sam2Processor,Sam2Model
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    model_path=Path(model_path);manifest=model_path/'download_manifest.json'
    identity=dict(image_sha256=sha256_file(path),boxes=boxes,model_manifest_sha256=sha256_file(manifest),
        code_sha256=sha256_file(__file__),private_gallery_used=False)
    if(root/'record.json').exists():
        old=read_json(root/'record.json')
        if old['identity']!=identity or sha256_file(root/'foreground.png')!=old['mask_sha256']:raise ValueError('分割来源改变')
        return read_rgb(root/'foreground.png')[...,0]>0,old
    if any(root.iterdir()):raise ValueError('拒绝覆盖未知分割目录')
    tick=time.perf_counter();image=Image.open(relocate_io_path(path)).convert('RGB')
    processor=Sam2Processor.from_pretrained(str(model_path),local_files_only=True)
    model,info=Sam2Model.from_pretrained(str(model_path),local_files_only=True,output_loading_info=True)
    if info.get('missing_keys') or info.get('mismatched_keys'):raise ValueError('分割模型缺失图像权重')
    model.eval();inputs=processor(images=image,input_boxes=[boxes],return_tensors='pt')
    with torch.no_grad():prediction=model(**inputs,multimask_output=True)
    masks=processor.post_process_masks(prediction.pred_masks,inputs['original_sizes'])[0]
    scores=prediction.iou_scores[0].cpu().numpy()
    masks=masks.cpu().numpy();choices,center_flags=foreground_choices(masks,scores,boxes)
    mask=np.logical_or.reduce([masks[i,choices[i]]for i in range(len(boxes))])
    if mask.sum()<image.width*image.height*.02:raise ValueError('主体分割为空或过小')
    Image.fromarray(mask.astype(np.uint8)*255).save(root/'foreground.png')
    overlay=np.asarray(image).copy();overlay[mask]=np.rint(.6*overlay[mask]+.4*np.array([240,40,50])).astype(np.uint8)
    Image.fromarray(overlay).save(root/'overlay.png')
    record=dict(identity=identity,seconds=time.perf_counter()-tick,pixels=int(mask.sum()),scores=scores.tolist(),
        mask_sha256=sha256_file(root/'foreground.png'),selection='maximum_model_iou_among_masks_containing_owned_box_center',
        selected_indices=choices,candidate_center_flags=center_flags)
    write_json(root/'record.json',record);del model,processor;gc.collect();return mask,record
