"""公开原图的文字区域与前景标注；自动提议后须有实际视觉核对记录。"""
from src.paths import ProjectPath as Path
import importlib.metadata
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

from src.paths import MODEL_ROOT
from src.visual.audit import digest, freeze_json, read_json, save_json
from src.visual.image_utils import sha256_file

ANNOTATION_POLICY = 'public_ocr_sam_positive_seed_background_connectivity_v2'


def regions_from_ocr(result, width, height, padding=3):
    regions, texts = [], []
    for quad, text, confidence in result or []:
        polygon = np.asarray(quad, dtype=float)
        if polygon.shape != (4, 2) or not np.isfinite(polygon).all():
            raise ValueError('文字识别区域无效')
        x0, y0 = np.maximum(polygon.min(axis=0)-padding, (0, 0))
        x1, y1 = np.minimum(polygon.max(axis=0)+padding, (width, height))
        if x1 > x0 and y1 > y0:
            regions.append([float(x0/width), float(y0/height), float(x1/width), float(y1/height)])
            texts.append(dict(text=str(text), confidence=float(confidence), polygon=polygon.tolist()))
    return regions, texts


def object_prior(image):
    """仅使用当前公开图片推断包围框与前景点，不使用商品类目或受害信号。"""
    array = np.asarray(image.convert('RGB'), dtype=np.float32)
    h, w = array.shape[:2]
    sh, sw = max(2, h//25), max(2, w//25)
    corners = np.concatenate([array[:sh,:sw].reshape(-1,3),array[:sh,-sw:].reshape(-1,3),
                              array[-sh:,:sw].reshape(-1,3),array[-sh:,-sw:].reshape(-1,3)])
    background = np.median(corners, axis=0)
    if np.quantile(np.max(np.abs(corners-background),axis=1), .75) > 45:
        raise ValueError('公开图片背景不均匀，需要单独核对主体标注')
    rough = np.max(np.abs(array-background),axis=2) > 18
    components, count = ndi.label(rough)
    sizes = np.bincount(components.ravel()); sizes[0] = 0
    good = sizes >= max(12, int(h*w*.0008))
    rough = good[components]
    if rough.sum() < h*w*.01:
        raise ValueError('公开图片没有足够的主体颜色线索')
    ys,xs = np.where(rough)
    pad_x,pad_y = max(2,int(w*.015)),max(2,int(h*.015))
    box = [max(0,int(xs.min())-pad_x),max(0,int(ys.min())-pad_y),
           min(w-1,int(xs.max())+pad_x),min(h-1,int(ys.max())+pad_y)]
    distance = ndi.distance_transform_edt(rough)
    sy,sx = np.unravel_index(np.argmax(distance), distance.shape)
    return rough,box,(int(sx),int(sy))


def remove_connected_background(image,mask):
    """去除与图像边界连通的近背景色，保留被物体边缘包围的亮部。"""
    array=np.asarray(image.convert('RGB'),dtype=np.float32)
    h,w=array.shape[:2];sh,sw=max(2,h//25),max(2,w//25)
    corners=np.concatenate([array[:sh,:sw].reshape(-1,3),array[:sh,-sw:].reshape(-1,3),
                             array[-sh:,:sw].reshape(-1,3),array[-sh:,-sw:].reshape(-1,3)])
    background=np.median(corners,axis=0)
    similar=np.max(np.abs(array-background),axis=2)<=12
    seeds=np.zeros_like(similar);seeds[0]=similar[0];seeds[-1]=similar[-1]
    seeds[:,0]|=similar[:,0];seeds[:,-1]|=similar[:,-1]
    connected=ndi.binary_propagation(seeds,mask=similar)
    return np.asarray(mask,dtype=bool)&~connected


class PublicImageAnnotator:
    def __init__(self):
        import torch
        import cv2
        from rapidocr_onnxruntime import RapidOCR
        import rapidocr_onnxruntime
        from transformers import Sam2Processor,Sam2Model
        self.torch = torch
        torch.set_num_threads(4)
        cv2.setNumThreads(1)
        self.ocr = RapidOCR(intra_op_num_threads=4, inter_op_num_threads=1)
        model = MODEL_ROOT/'sam2.1-hiera-tiny-source'
        self.processor=Sam2Processor.from_pretrained(str(model),local_files_only=True)
        self.model,info=Sam2Model.from_pretrained(str(model),local_files_only=True,output_loading_info=True)
        if info.get('missing_keys') or info.get('mismatched_keys'):
            raise ValueError('公开分割器缺失实际权重')
        self.model.eval().requires_grad_(False)
        package_root=Path(rapidocr_onnxruntime.__file__).parent
        self.snapshot=dict(policy=ANNOTATION_POLICY,ocr_version=importlib.metadata.version('rapidocr-onnxruntime'),
                           ocr_weights={p.name:sha256_file(p) for p in package_root.rglob('*.onnx')},
                           segmentation_weights_sha256=sha256_file(model/'model.safetensors'),
                           segmentation_model_manifest_sha256=sha256_file(model/'download_manifest.json'),
                           code_sha256=sha256_file(__file__),device='cpu',model_loaded_once=True)

    def propose(self,record,output):
        output=Path(output);output.mkdir(parents=True,exist_ok=True)
        identity=dict(image_sha256=record['image_sha256'],annotation_models=self.snapshot)
        final=output/'proposal.json'
        if final.exists():
            saved=read_json(final)
            if saved['identity']!=identity or sha256_file(saved['foreground_mask'])!=saved['foreground_mask_sha256']:
                raise ValueError('公开标注缓存身份改变')
            return saved
        started=time.perf_counter()
        image=Image.open(record['image']).convert('RGB')
        result,_=self.ocr(record['image'],use_cls=False)
        regions,texts=regions_from_ocr(result,image.width,image.height)
        rough,box,seed=object_prior(image)
        inputs=self.processor(images=image,input_boxes=[[box]],input_points=[[[list(seed)]]],
                              input_labels=[[[1]]],return_tensors='pt')
        with self.torch.no_grad():prediction=self.model(**inputs,multimask_output=True)
        masks=self.processor.post_process_masks(prediction.pred_masks,inputs['original_sizes'])[0].cpu().numpy()[0]
        scores=prediction.iou_scores[0,0].cpu().numpy()
        candidates=[]
        masks=np.stack([remove_connected_background(image,m) for m in masks])
        for i,mask in enumerate(masks):
            ratio=float(mask.mean());coverage=float((mask&rough).sum()/rough.sum())
            accepted=bool(mask[seed[1],seed[0]] and .01<=ratio<=.97 and coverage>=.30)
            candidates.append(dict(index=i,iou=float(scores[i]),foreground_ratio=ratio,rough_coverage=coverage,accepted=accepted))
        eligible=[r for r in candidates if r['accepted']]
        if not eligible:
            save_json(output/'rejection.json',dict(identity=identity,box=box,seed=seed,candidates=candidates))
            raise ValueError('公开主体分割没有通过固定的前景质量门槛')
        chosen=max(eligible,key=lambda r:(r['iou'],-r['index']))
        foreground=masks[chosen['index']]
        path=output/'foreground.png';Image.fromarray(foreground.astype(np.uint8)*255).save(path)
        ys,xs=np.where(foreground)
        boxes=[[int(xs.min()),int(ys.min()),min(image.width-1,int(xs.max())),min(image.height-1,int(ys.max()))]]
        overlay=np.asarray(image).copy()
        edge=foreground&~ndi.binary_erosion(foreground,iterations=2)
        overlay[edge]=[0,180,40]
        display=Image.fromarray(overlay);draw=ImageDraw.Draw(display)
        for x0,y0,x1,y1 in regions:draw.rectangle((x0*image.width,y0*image.height,x1*image.width,y1*image.height),outline='red',width=2)
        display.save(output/'review_overlay.png')
        proposal=dict(id=record['id'],identity=identity,image=record['image'],title=record['title'],
                      foreground_mask=str(path.resolve()),foreground_mask_sha256=sha256_file(path),
                      boxes=boxes,text_regions=regions,ocr=texts,segmentation_candidates=candidates,
                      selected_mask_index=chosen['index'],text_regions_reviewed=False,
                      review_overlay=str((output/'review_overlay.png').resolve()),
                      observed_seconds=time.perf_counter()-started)
        freeze_json(final,proposal)
        return proposal

    def close(self):
        self.ocr=self.model=self.processor=None
        import gc
        gc.collect()


def annotate_downloads(downloads,output):
    output=Path(output)
    final=output/'annotation_manifest.json'
    annotator=PublicImageAnnotator()
    proposals,failures={},{}
    try:
        for pid,record in sorted(downloads['records'].items()):
            try:proposals[pid]=annotator.propose(record,output/'annotations'/pid)
            except Exception as error:failures[pid]=dict(type=type(error).__name__,reason=str(error))
            print(f'公开原图标注 {len(proposals)+len(failures)}/{len(downloads["records"])}，失败 {len(failures)}',flush=True)
        result=dict(downloads_sha256=digest(downloads),proposals=proposals,failures=failures,models=annotator.snapshot)
        freeze_json(final,result)
    finally:annotator.close()
    return result


def contact_sheets(shortlist,annotations,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15) if Path('C:/Windows/Fonts/arial.ttf').is_file() else ImageFont.load_default()
    sheets=[]
    for group in shortlist['groups']:
        items=[group['target']]+group['candidates']
        items=[p for p in items if p['id'] in annotations['proposals']]
        for page in range(0,len(items),6):
            batch=items[page:page+6]
            canvas=Image.new('RGB',(900,3*510),'#e8e8e8');draw=ImageDraw.Draw(canvas)
            for i,product in enumerate(batch):
                proposal=annotations['proposals'][product['id']]
                image=Image.open(proposal['review_overlay']).convert('RGB');image.thumbnail((430,430))
                x=(i%2)*450+(450-image.width)//2;y=(i//2)*510+55
                canvas.paste(image,(x,y))
                role='TARGET' if product['id']==group['target']['id'] else 'SOURCE'
                draw.text(((i%2)*450+8,(i//2)*510+5),f"{role} {product['id']}",fill='black',font=font)
                draw.text(((i%2)*450+8,(i//2)*510+25),product['title'][:48],fill='black',font=font)
            path=output/f"{group['key']}_{page//6+1:02d}.jpg";canvas.save(path,quality=92)
            sheets.append(dict(group=group['key'],path=str(path.resolve()),ids=[p['id'] for p in batch]))
    save_json(output/'sheets.json',sheets)
    return sheets


def apply_reviews(shortlist,downloads,annotations,reviews,output):
    """实际核对记录与图像、自动标注摘要绑定后，输出核心筛选器接受的清单。"""
    output=Path(output)
    if not isinstance(reviews,dict) or not str(reviews.get('reviewer','')).strip():
        raise ValueError('需要实际公开图片视觉核对者和逐图记录')
    if reviews.get('annotation_manifest_sha256')!=digest(annotations):
        raise ValueError('视觉核对没有绑定当前自动标注')
    rows=reviews.get('records',{})
    catalogs,exclusions={},{}
    for group in shortlist['groups']:
        prepared=[]
        for product in [group['target']]+group['candidates']:
            pid=product['id'];is_target=pid==group['target']['id']
            if pid not in annotations['proposals']:
                exclusions[pid]='download_or_annotation_failed'
                if is_target:raise ValueError('固定目标缺少有效公开图标注')
                continue
            proposal=annotations['proposals'][pid];review=rows.get(pid,{})
            if (review.get('image_sha256')!=downloads['records'][pid]['image_sha256']
                    or review.get('proposal_sha256')!=digest(proposal)):
                raise ValueError('缺少当前图片与标注的实际视觉核对：'+pid)
            if review.get('accepted') is not True:
                exclusions[pid]=review.get('reason','public_visual_quality_rejected')
                if is_target:raise ValueError('固定目标未通过公开图像核对')
                continue
            regions=review.get('text_regions',proposal['text_regions'])
            record=dict(id=pid,title=product['title'],image=downloads['records'][pid]['image'],
                        source_url=product['source_url'],boxes=review.get('boxes',proposal['boxes']),
                        foreground_mask=proposal['foreground_mask'],text_regions=regions,text_regions_reviewed=True)
            if not is_target:
                record.update(owned=True,source_note='固定散列规则预先登记的实验模拟可控商品；不声明真实商家所有权')
            prepared.append(record)
        if not prepared or prepared[0]['id']!=group['target']['id'] or len(prepared)<2:
            raise ValueError('公开质量检查后没有可用源商品：'+group['key'])
        target,candidates=prepared[0],prepared[1:]
        seen={downloads['records'][target['id']]['image_sha256']}
        distinct=[]
        for product in sorted(candidates,key=lambda p:p['id']):
            image_hash=downloads['records'][product['id']]['image_sha256']
            if image_hash in seen:
                exclusions[product['id']]='duplicate_public_main_image'
            else:seen.add(image_hash);distinct.append(product)
        if not distinct:raise ValueError('公开主图去重后没有源商品')
        catalog=dict(schema_version=1,target=target,candidates=distinct)
        path=output/'catalogs'/(group['key']+'.json');freeze_json(path,catalog)
        catalogs[group['key']]=str(path.resolve())
    result=dict(catalogs=catalogs,exclusions=exclusions,reviews_sha256=digest(reviews),
                annotations_sha256=digest(annotations),public_only=True)
    freeze_json(output/'reviewed_catalogs.json',result)
    return result
