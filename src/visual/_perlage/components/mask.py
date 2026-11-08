"""源图印刷墨迹保护与底色释放；不读取目标图库或检索名次。"""

from src.paths import relocate_io_path

from dataclasses import asdict,dataclass

from src.paths import ProjectPath as Path

import json

import numpy as np

from PIL import Image,ImageDraw,ImageFont

from scipy.ndimage import grey_closing,binary_dilation,distance_transform_edt

from src.visual.image_utils import sha256_file,write_json





@dataclass(frozen=True)

class SurfaceMaskConfig:

    closing_pixels:int=9

    local_ink_contrast:float=3.

    dark_luminance:float=180.

    colored_chroma:float=12.

    ink_margin_pixels:int=1

    feather_pixels:float=2.



    def validate(self):

        if self.closing_pixels<3 or self.closing_pixels%2!=1:raise ValueError('背景估计窗口应为至少三的奇数')

        if not 0<self.local_ink_contrast<32 or not 0<self.dark_luminance<255:raise ValueError('墨迹阈值无效')

        if not 0<self.colored_chroma<128 or not 0<=self.ink_margin_pixels<=4 or not 0<self.feather_pixels<=8:

            raise ValueError('保护边距或羽化宽度无效')





def region_mask(shape,regions):

    h,w=shape[:2];mask=np.zeros((h,w),bool)


    for x0,y0,x1,y1 in regions:

        if not 0<=x0<x1<=1 or not 0<=y0<y1<=1:raise ValueError('印刷区域必须是有效归一化矩形')

        mask[round(y0*h):round(y1*h),round(x0*w):round(x1*w)]=True

    return mask





def surface_masks(source,regions,config=SurfaceMaskConfig()):

    config.validate();a=np.asarray(source,dtype=np.float64)

    if a.ndim!=3 or a.shape[-1]!=3 or not np.isfinite(a).all()or a.min()<0 or a.max()>255:

        raise ValueError('源图必须是有效的原尺寸三通道八位像素')

    roi=region_mask(a.shape,regions)

    luminance=a@np.array([.2126,.7152,.0722])

    background=grey_closing(luminance,size=(config.closing_pixels,config.closing_pixels),mode='nearest')

    chroma=a.max(-1)-a.min(-1)

    ink=roi&((background-luminance>=config.local_ink_contrast)|(luminance<=config.dark_luminance)|(chroma>=config.colored_chroma))

    for region in regions:
        local=region_mask(a.shape,[region])
        if np.median(luminance[local])<128:ink|=local
    if roi.any()and not ink.any():raise ValueError('未检出印刷墨迹，拒绝把空掩码当内容保护')

    core=binary_dilation(ink,iterations=config.ink_margin_pixels)if config.ink_margin_pixels else ink.copy()

    edit=(np.rint(np.minimum(distance_transform_edt(~core)/config.feather_pixels,1.)*255).astype(np.uint8)
          if core.any()else np.full(roi.shape,255,dtype=np.uint8))

    if np.any(edit[core]!=0):raise AssertionError('墨迹核心应禁止编辑')

    return dict(edit=edit,ink=ink,core=core,legacy_region=roi,background=background)





def build_surface_masks(source,regions,output,config=SurfaceMaskConfig()):

    source=Path(source).resolve();root=Path(output);root.mkdir(parents=True,exist_ok=True)

    identity=json.loads(json.dumps(dict(source_sha256=sha256_file(source),regions=regions,config=asdict(config),

        implementation_sha256=sha256_file(__file__),selection_uses_source_only=True,content_semantics_certified=False)))

    ip=root/'identity.json'

    if ip.exists()and json.loads(ip.read_text(encoding='utf-8'))!=identity:raise ValueError('掩码来源或实现改变，请另建版本')

    if not ip.exists()and any(root.iterdir()):raise ValueError('拒绝覆盖未知掩码目录')

    with Image.open(relocate_io_path(source))as im:a=np.asarray(im.convert('RGB')).copy()

    masks=surface_masks(a,regions,config)

    files={}

    for name in('edit','ink','core','legacy_region'):

        raw=masks[name]if name=='edit'else masks[name].astype(np.uint8)*255

        path=root/(name+'.png');Image.fromarray(raw).save(path);files[name]=dict(path=str(path.resolve()),sha256=sha256_file(path))

    record=dict(identity=identity,files=files,legacy_protected_pixels=int(masks['legacy_region'].sum()),

        protected_ink_pixels=int(masks['ink'].sum()),protected_core_pixels=int(masks['core'].sum()),

        released_legacy_pixels=int((masks['legacy_region']&~masks['core']).sum()),

        source_derived_candidate_mask=True,requires_manual_content_and_logo_review=True,

        caveat='Source contrast identifies candidate printed marks, not semantic recognition or proof that every character is covered.')

    write_json(ip,identity);write_json(root/'mask_record.json',record)

    h,w=a.shape[:2];panel=Image.new('RGB',(3*w,h+54),'white');draw=ImageDraw.Draw(panel)

    try:font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',20)

    except OSError:font=ImageFont.load_default()

    overlay=a.copy();core=masks['core'];overlay[core]=np.rint(.4*overlay[core]+.6*np.array([220,30,60])).astype(np.uint8)

    for i,(label,im)in enumerate([('原照片',Image.fromarray(a)),('墨迹与商标保护候选',Image.fromarray(overlay)),('白色可编辑，黑色保护',Image.fromarray(masks['edit']).convert('RGB'))]):

        panel.paste(im,(i*w,54));draw.text((i*w+8,12),label,font=font,fill='black')

    panel.save(root/'mask_comparison.png');return record





def legacy_boundary_residual_jump(source,candidate,regions):

    """在原矩形边界、源图相邻平坦亮像素对上比较残差跳变；仅是拼接诊断。"""

    a,b=np.asarray(source,dtype=float),np.asarray(candidate,dtype=float)

    if a.shape!=b.shape:raise ValueError('边界诊断图片尺寸不同')

    h,w=a.shape[:2];pairs=set()

    for x0,y0,x1,y1 in regions:

        left,right,top,bottom=round(x0*w),round(x1*w),round(y0*h),round(y1*h)

        for x in range(left,right):

            for y in(top,bottom):

                if 0<y<h:pairs.add((y-1,x,y,x))

        for y in range(top,bottom):

            for x in(left,right):

                if 0<x<w:pairs.add((y,x-1,y,x))

    residual=b-a;values=[]

    for y0,x0,y1,x1 in sorted(pairs):

        u,v=a[y0,x0],a[y1,x1]

        if min(u.min(),v.min())>=210 and max(np.ptp(u),np.ptp(v))<=8 and np.max(np.abs(u-v))<=5:

            values.append(np.abs(residual[y0,x0]-residual[y1,x1]).mean())

    if not values:raise ValueError('没有适用于该边界诊断的源图平坦亮像素对')

    return dict(source_fixed_pairs=len(values),mean_residual_jump=float(np.mean(values)),

        p95_residual_jump=float(np.quantile(values,.95)),not_a_global_naturalness_metric=True)
