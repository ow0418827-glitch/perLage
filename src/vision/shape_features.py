"""冻结小型视觉模型的整图灰度全局特征；不分割、不识字、不试改色。"""
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
from src.vision.audit import digest,read_json,freeze_json,model_files
from src.vision.image_utils import sha256_file

PREPROCESSING=dict(policy='whole_image_gray_letterbox224_global_cls_v2',size=224,
    grayscale='Pillow_L_then_three_channels',resize='bicubic_preserve_aspect',padding=127,
    foreground_segmentation=False,text_detection=False,pilot_edit=False)


def shape_input(path):
    with Image.open(path) as opened:
        gray=ImageOps.exif_transpose(opened).convert('L')
        contained=ImageOps.contain(gray,(224,224),Image.Resampling.BICUBIC)
    canvas=Image.new('L',(224,224),127)
    canvas.paste(contained,((224-contained.width)//2,(224-contained.height)//2))
    return canvas.convert('RGB')


class LocalShapeBackend:
    def __init__(self,settings,output,device='cuda'):
        if set(settings)-{'dino_model','batch_size'}:raise ValueError('新选图器只接受一个冻结视觉模型及批量大小')
        from src.paths import MODEL_ROOT
        from transformers import AutoImageProcessor,AutoModel
        import transformers
        import torch
        import PIL
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.device=device;self.batch_size=settings.get('batch_size',16)
        if type(self.batch_size) is not int or self.batch_size<1:raise ValueError('批量大小须为正整数')
        path=Path(settings.get('dino_model',MODEL_ROOT/'dinov2-small')).resolve()
        self.processor=AutoImageProcessor.from_pretrained(path,local_files_only=True)
        self.model=AutoModel.from_pretrained(path,local_files_only=True).float().eval().requires_grad_(False).to(device)
        self.snapshot=dict(dino=model_files(path),preprocessing=PREPROCESSING,precision='float32',
            libraries=dict(transformers=transformers.__version__,torch=torch.__version__,pillow=PIL.__version__,numpy=np.__version__),
            implementation_sha256=sha256_file(__file__))
        self.identity=digest(self.snapshot);self.cache={}

    def features_many(self,products):
        import torch
        pending={}
        for product in products:
            actual=sha256_file(product['image'])
            if actual!=product['image_sha256']:raise ValueError('公开图片内容与候选记录不一致')
            identity=dict(image_sha256=actual,model_sha256=self.identity,preprocessing=PREPROCESSING)
            key=digest(identity);path=self.output/(key+'.json')
            if key in self.cache:continue
            if path.exists():
                saved=read_json(path)
                if saved['identity']!=identity or saved['vector_sha256']!=digest(saved['vector']):raise ValueError('外形特征缓存身份或内容改变')
                vector=np.asarray(saved['vector'],dtype=np.float64)
                if vector.ndim!=1 or not np.isfinite(vector).all() or np.linalg.norm(vector)==0:raise ValueError('外形特征缓存无效')
                self.cache[key]=dict(shape=vector)
            else:pending[key]=(product,identity,path)
        items=list(pending.items())
        for start in range(0,len(items),self.batch_size):
            batch=items[start:start+self.batch_size]
            images=[shape_input(value[0]['image']) for _,value in batch]
            inputs=self.processor(images=images,do_resize=False,do_center_crop=False,return_tensors='pt').to(self.device)
            with torch.inference_mode():vectors=self.model(**inputs).last_hidden_state[:,0,:].float().cpu().numpy()
            for (key,(_,identity,path)),vector in zip(batch,vectors):
                values=vector.astype(np.float64).tolist()
                freeze_json(path,dict(identity=identity,vector=values,vector_sha256=digest(values)))
                self.cache[key]=dict(shape=np.asarray(values))

    def features(self,product):
        self.features_many([product])
        key=digest(dict(image_sha256=product['image_sha256'],model_sha256=self.identity,preprocessing=PREPROCESSING))
        return self.cache[key]

    def close(self):
        self.model=None;self.cache.clear()
        import gc
        gc.collect()
        if self.device.startswith('cuda'):
            import torch
            torch.cuda.empty_cache()
