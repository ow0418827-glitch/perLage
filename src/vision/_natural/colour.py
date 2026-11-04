"""主体前景分割与统计选色；唯一自然优先流程的组成步骤。"""
from src.paths import relocate_io_path
from src.paths import ProjectPath as Path
from dataclasses import dataclass,asdict
import gc,time,json
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import grey_closing,gaussian_filter,distance_transform_edt,binary_erosion,binary_propagation
from skimage.color import rgb2lab,lab2rgb
from src.vision.image_utils import sha256_file,write_json
from src.vision._natural.assets import read_json


def linear(a):
    a=np.asarray(a,dtype=np.float64)
    return np.where(a<=.04045,a/12.92,((a+.055)/1.055)**2.4)


def srgb(a):
    a=np.clip(a,0,1)
    return np.where(a<=.0031308,a*12.92,1.055*np.power(a,1/2.4)-.055)


def read_rgb(path):return np.asarray(Image.open(relocate_io_path(path)).convert('RGB')).copy()






@dataclass(frozen=True)
class HarmonizationConfig:
    smoothing_sigma:float=12.
    print_ring_sigma:float=7.
    selected_color_mix:float=.25
    strength:float=1.
    blend_mode:str='matte'

    def validate(self):
        if not 1<=self.smoothing_sigma<=32 or not 1<=self.print_ring_sigma<=16:raise ValueError('空间尺度无效')
        if not 0<=self.selected_color_mix<=1 or not 0<self.strength<=1:raise ValueError('颜色混合强度无效')
        if self.blend_mode not in ('matte','chroma'):raise ValueError('未知融合方式')


def print_region_mask(shape, regions):
    result=np.zeros(shape[:2],dtype=bool)
    h,w=result.shape
    for x0,y0,x1,y1 in regions:
        if not 0<=x0<x1<=1 or not 0<=y0<y1<=1:raise ValueError('印刷区域无效')
        result[round(y0*h):round(y1*h),round(x0*w):round(x1*w)]=True
    return result


def foreground_statistics(target, target_foreground):
    """沿用源最终方案的主体背景排除、亮度分位裁剪及颜色空间均值。"""
    target_array=read_rgb(target);lab=rgb2lab(target_array/255.)
    border=np.concatenate([target_array[0],target_array[-1],target_array[:,0],target_array[:,-1]],axis=0)
    background_color=np.median(border,axis=0)
    potential=np.linalg.norm(target_array.astype(float)-background_color,axis=-1)<=10
    seed=np.zeros(potential.shape,bool);seed[[0,-1],:]=potential[[0,-1],:];seed[:,[0,-1]]=potential[:,[0,-1]]
    background=binary_propagation(seed,mask=potential)
    fg=binary_erosion(target_foreground&~background,iterations=2)
    if fg.sum()<100:raise ValueError('目标主体有效像素不足')
    values=lab[fg];low,high=np.quantile(values[:,0],[.20,.90])
    body=fg&(lab[...,0]>=low)&(lab[...,0]<=high)
    if body.sum()<100:raise ValueError('目标主体有效色彩像素不足')
    distribution=lab[body];mean=distribution.mean(0);cov=np.cov(distribution.T)
    rgb=np.rint(lab2rgb(mean.reshape(1,1,3))[0,0]*255).astype(int).tolist()
    return dict(color_scheme='statistics',rgb=rgb,target_foreground_pixels=int(fg.sum()),
        body_pixels=int(body.sum()),luminance_quantiles=[float(low),float(high)],
        photo_border_background_rgb=background_color.tolist(),
        border_connected_background_pixels=int(background.sum()),lab_mean=mean.tolist(),
        lab_covariance=cov.tolist(),covariance_diagnostic_only=True),body


def _prepare_general_context(source,parent_anchor,parent_edit_mask,foreground,regions):
    """同一流程兼容无印刷商品及深色衣料，不把瓶身白底假设用于所有商品。"""
    source=np.asarray(source,dtype=np.uint8);parent_anchor=np.asarray(parent_anchor,dtype=np.uint8)
    if source.shape!=parent_anchor.shape or foreground.shape!=source.shape[:2]or parent_edit_mask.shape!=foreground.shape:
        raise ValueError('源图、锚点和掩码形状不同')
    roi=print_region_mask(source.shape,regions)
    src=linear(source/255.);parent=linear(parent_anchor/255.)
    weights=np.array([.2126,.7152,.0722]);ys=src@weights
    paper=np.stack([grey_closing(src[...,c],size=(11,11),mode='nearest')for c in range(3)],-1)
    yb=paper@weights
    transmission=np.zeros(source.shape[:2],float);opaque=np.zeros_like(roi)
    colored=(source.max(-1).astype(float)-source.min(-1))>=12
    for region in regions:
        local=print_region_mask(source.shape,[region]);inside=local&foreground
        if not inside.any():continue
        # 深色织物的白色及彩色印字连同底色保守保留；不套用白纸上的深墨估计。
        if np.median(ys[inside])<.35:
            opaque|=local
        else:
            darkest=float(np.quantile(ys[inside],.02))
            trans=np.clip((ys-darkest)/np.maximum(yb-darkest,1e-4),0,1)
            local_opaque=local&((trans<=.04)|colored)
            transmission[local]=trans[local];opaque|=local_opaque
    transmission[opaque]=0
    ink=roi&((source.min(-1)<180)|colored)
    distance=distance_transform_edt(~ink)if ink.any()else np.full(roi.shape,64.)
    trusted=foreground&~roi&(parent_edit_mask>=254)
    if trusted.sum()<100:raise ValueError('未印刷主体参考像素不足')
    halo=roi&foreground&(parent_edit_mask==0)&(source.min(-1)>=220)&(transmission>.85)
    _,nearest=distance_transform_edt(~trusted,return_indices=True)
    return dict(source=source,parent=parent,parent_u8=parent_anchor,delta=parent-src,
        transmission=transmission,foreground=foreground,roi=roi,opaque=opaque,trusted=trusted,
        halo=halo,halo_applicable=bool(halo.any()),nearest=nearest,distance=distance)


def halo_error(context,image):
    if not context['halo'].any():return 0.
    difference=np.asarray(image,dtype=float)-context['source'].astype(float)
    h=context['halo'];ny,nx=context['nearest']
    return float(np.linalg.norm(difference[h]-difference[ny[h],nx[h]],axis=-1).mean())




from src.vision._natural.segmentation import segment_owned_image,foreground_choices

from src.vision._natural.mask import surface_masks,SurfaceMaskConfig

def _prepare_light_context(source,parent_anchor,parent_edit_mask,foreground,regions):
    source=np.asarray(source,dtype=np.uint8);parent_anchor=np.asarray(parent_anchor,dtype=np.uint8)
    if source.shape!=parent_anchor.shape or foreground.shape!=source.shape[:2] or parent_edit_mask.shape!=foreground.shape:
        raise ValueError('源图、锚点和掩码形状不同')
    raw=surface_masks(source,regions,SurfaceMaskConfig());roi=raw['legacy_region']
    src=linear(source/255.);parent=linear(parent_anchor/255.)
    paper=np.stack([grey_closing(src[...,c],size=(11,11),mode='nearest')for c in range(3)],-1)
    weights=np.array([.2126,.7152,.0722]);ys=src@weights;yb=paper@weights
    darkest=float(np.quantile(ys[roi],.02))
    transmission=np.clip((ys-darkest)/np.maximum(yb-darkest,1e-4),0,1)
    colored=(source.max(-1).astype(float)-source.min(-1))>=12
    transmission[colored&roi]=0
    # 真正深色核心保持源色；灰色抗锯齿边缘可随纸底重合成。
    opaque=roi&((transmission<=.04)|colored)
    transmission[opaque]=0
    distance=distance_transform_edt(~raw['ink'])
    trusted=foreground&(parent_edit_mask>=254)&(source.min(-1)>=210)&(distance>=4)
    if trusted.sum()<100:raise ValueError('未印刷表面参考点不足')
    halo=roi&foreground&(parent_edit_mask==0)&(source.min(-1)>=220)&(transmission>.85)
    if not halo.any():raise ValueError('未找到源图定义的可诊断纸底光晕')
    _,nearest=distance_transform_edt(~trusted,return_indices=True)
    return dict(source=source,parent=parent,parent_u8=parent_anchor,delta=parent-src,transmission=transmission,
        foreground=foreground,roi=roi,opaque=opaque,trusted=trusted,halo=halo,nearest=nearest,distance=distance)


def harmonize_statistics(context,parent_edit_mask,color_rgb,config=HarmonizationConfig()):
    config.validate()
    if config.blend_mode!='matte':raise ValueError('细节保留分支仅采用线性光底色重合成')
    c=context;weight=c['trusted'].astype(float)
    den=gaussian_filter(weight,config.smoothing_sigma,mode='nearest')
    smooth=np.stack([gaussian_filter(c['delta'][...,i]*weight,config.smoothing_sigma,mode='nearest')/np.maximum(den,1e-6)for i in range(3)],-1)
    smooth[den<.005]=np.median(c['delta'][c['trusted']],axis=0)
    direction=linear(np.asarray(color_rgb,dtype=float)/255)-1
    luminance=np.array([.2126,.7152,.0722]);dy=smooth@luminance;direction_y=float(direction@luminance)
    selected=smooth.copy()if direction_y>=-1e-5 else np.clip(dy/direction_y,0,1)[...,None]*direction
    desired=(1-config.selected_color_mix)*smooth+config.selected_color_mix*selected
    ring=np.exp(-.5*(c['distance']/config.print_ring_sigma)**2)*c['roi']*c['foreground']
    missing=1-np.asarray(parent_edit_mask,dtype=float)/255
    # 旧方法重建整个印刷区会减掉已有生成残差。此处仅补尚未编辑的覆盖份额。
    correction=config.strength*ring[...,None]*missing[...,None]*c['transmission'][...,None]*desired
    result=np.rint(srgb(c['parent']+correction)*255).astype(np.uint8)
    fixed=(parent_edit_mask==255)|(ring<=1e-5)
    result[fixed]=c['parent_u8'][fixed]
    result[c['opaque']]=c['source'][c['opaque']]
    if np.any(result[parent_edit_mask==255]!=c['parent_u8'][parent_edit_mask==255]):
        raise AssertionError('已经生成的可编辑表面被改写')
    return result


def prepare_print_context(source,parent_anchor,parent_edit_mask,foreground,regions):
    # 原样保留源最终方案的白色印刷底色计算；无纸底或深色织物仅扩展输入有效性。
    generic=_prepare_general_context(source,parent_anchor,parent_edit_mask,foreground,regions)
    if generic['halo_applicable']:
        original=_prepare_light_context(source,parent_anchor,parent_edit_mask,foreground,regions)
        original['halo_applicable']=True
        return original
    return generic
