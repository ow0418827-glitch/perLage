"""真实潜空间扩散净化及其自适应梯度研究组件。

使用本地 SD1.5(稳定扩散第一版修订版)的冻结权重、空提示、确定性隐式
扩散逆过程与随机前向噪声。不是原版 DiffPure(扩散对抗净化)或认知守卫
完整复现；不包含文字裁判、良性参考停止、多数投票拒绝或外部网络请求。

完整梯度使用逐步检查点重计算，固定同一前向噪声与时间步，不调用旧攻击
模块中停止梯度的噪声预测函数。近似梯度仅为消融，不能用于证明防御安全。
"""
from __future__ import annotations
from src.paths import model_path

from dataclasses import asdict, dataclass
from src.paths import ProjectPath as Path
import hashlib

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.paths import PROJECT


@dataclass(frozen=True)
class PurificationConfig:
    model_path: str = model_path('stable-diffusion-v1-5')
    working_size: int = 256
    reverse_steps: int = 10
    start_timestep: int = 99
    gradient_checkpointing: bool = True

    def validate(self):
        if not isinstance(self.model_path,(str,Path)) or not str(self.model_path).strip():
            raise ValueError('净化模型目录必须为非空本地路径')
        for key in ('working_size','reverse_steps','start_timestep'):
            value=getattr(self,key)
            if isinstance(value,bool) or not isinstance(value,int):
                raise ValueError(f'{key} 必须为整数')
        if not 64 <= self.working_size <= 768 or self.working_size % 8:
            raise ValueError('净化工作长边必须为六十四至七百六十八内八的倍数')
        if not 1 <= self.start_timestep <= 999:
            raise ValueError('起始训练时间步必须为一至九百九十九')
        if not 2 <= self.reverse_steps <= self.start_timestep+1:
            raise ValueError('逆过程步数必须至少二步且不能产生重复时间步')
        if not isinstance(self.gradient_checkpointing,bool):
            raise ValueError('检查点开关必须为布尔值')


def working_shape(shape, size):
    height,width=shape[-2:]
    scale=size/max(height,width)
    return tuple(max(64,int(round(n*scale/8))*8) for n in (height,width))


def ddim_step(sample, noise, alpha, previous_alpha):
    """确定性隐式扩散一步；显式相邻时间步，避免非均匀步长索引错配。"""
    clean=(sample-(1-alpha).sqrt()*noise)/alpha.sqrt()
    return previous_alpha.sqrt()*clean+(1-previous_alpha).sqrt()*noise


def replace_backward(actual, proxy):
    """真实前向、代理反向；括号先做同值相减，避免改变真实前向舍入结果。"""
    return actual.detach()+(proxy-proxy.detach())


class LatentDiffusionPurifier:
    def __init__(self,config=PurificationConfig(),device='cuda',*,components=None):
        config.validate()
        self.config,self.device=config,torch.device(device)
        if self.device.type=='cuda' and self.device.index is None:
            self.device=torch.device('cuda',torch.cuda.current_device())
        if components is None:
            from src.vision.diffusion_components import load_diffusion_components
            components=load_diffusion_components({
                'model_path':config.model_path,'variant':'fp16','prompt':'',
                'prompt_guidance_scale':1.0,'attention_slicing':False},self.device)
        self.components=components
        self.unet,self.vae=components.unet,components.vae
        for model in (self.unet,self.vae):
            model.eval().requires_grad_(False)
        if components.scheduler.config.prediction_type != 'epsilon':
            raise ValueError('当前实现只接受噪声预测模型')
        self.alphas=components.scheduler.alphas_cumprod.detach().to(
            self.device,dtype=torch.float32).clone()
        if self.alphas.shape!=(1000,):
            raise ValueError('当前研究协议固定一千个训练时间步')
        if (not torch.isfinite(self.alphas).all() or not (self.alphas>0).all()
                or not (self.alphas<=1).all()
                or not (self.alphas[1:]<=self.alphas[:-1]).all()):
            raise ValueError('累计信号日程必须有限、位于零至一且单调不增')
        self.forward_noise_calls=0

    def snapshot(self):
        return {'kind':'sd15_latent_noise_then_ddim_purification_v1',
                'config':asdict(self.config),'model':self.components.manifest,
                'prompt':'','guidance_scale':1.0,'posterior':'mode','eta':0.0,
                'resize':'aspect_long_edge_round8_min64_bicubic_antialias_back_to_native',
                'output_range':[0.0,1.0],'quantization':'external_eight_bit_save_or_ste',
                'implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'not_original_diffpure':True,'not_full_cognitiveguard':True}

    def _validate_input(self,image):
        if image.ndim!=4 or image.shape[:2]!=(1,3) or min(image.shape[-2:])<8:
            raise ValueError('净化仅接受单张三通道图片批次')
        if image.device!=self.device or image.dtype!=torch.float32:
            raise ValueError('净化图片必须为指定设备上的单精度张量')
        if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError('净化图片必须有限且位于零至一')

    def _encode(self,image):
        work=F.interpolate(image,size=working_shape(image.shape,self.config.working_size),
                           mode='bicubic',align_corners=False,antialias=True).clamp(0,1)
        return self.vae.encode(work*2-1).latent_dist.mode().float()*self.vae.config.scaling_factor

    def _decode(self,latent,shape):
        image=(self.vae.decode(latent/self.vae.config.scaling_factor).sample.float()/2+0.5).clamp(0,1)
        return F.interpolate(image,size=shape[-2:],mode='bicubic',align_corners=False,antialias=True).clamp(0,1)

    def reconstruct(self,image):
        """无加噪、无噪声网络的编码解码对照，用于区分压缩与扩散作用。"""
        self._validate_input(image)
        with torch.autocast(device_type=self.device.type,enabled=False):
            result=self._decode(self._encode(image),image.shape)
        self._require_finite(result,'自编码器重建图片')
        return result

    @staticmethod
    def _require_finite(value,label):
        if not torch.isfinite(value).all():
            raise FloatingPointError(f'{label}出现非有限数值，不能保存为净化图片')

    def __call__(self,image,*,seed,start_timestep=None,gradient='none'):
        self._validate_input(image)
        if isinstance(seed,bool) or not isinstance(seed,int) or not 0<=seed<2**64:
            raise ValueError('净化种子必须是小于二的六十四次方的非负整数')
        start=self.config.start_timestep if start_timestep is None else start_timestep
        if isinstance(start,bool) or not isinstance(start,int) or not self.config.reverse_steps-1<=start<=999:
            raise ValueError('净化起点不能超出范围或产生重复时间步')
        if gradient not in ('none','identity','vae','full'):
            raise ValueError('未知净化梯度模式')
        if gradient in ('identity','vae'):
            with torch.no_grad():
                actual=self._run(image,seed,start)
            proxy=image if gradient=='identity' else self.reconstruct(image)
            return replace_backward(actual,proxy)
        if gradient=='none':
            with torch.no_grad():
                return self._run(image,seed,start)
        return self._run(image,seed,start)

    def _run(self,image,seed,start):
        # 外部混合精度上下文不能悄悄改变自编码器及扩散日程的单精度约定。
        with torch.autocast(device_type=self.device.type,enabled=False):
            return self._run_float32(image,seed,start)

    def _run_float32(self,image,seed,start):
        latent=self._encode(image)
        self._require_finite(latent,'编码潜变量')
        generator=torch.Generator(device=self.device).manual_seed(seed)
        noise=torch.randn(latent.shape,generator=generator,device=self.device,dtype=torch.float32)
        alpha=self.alphas[start]
        latent=alpha.sqrt()*latent+(1-alpha).sqrt()*noise
        times=torch.linspace(start,0,self.config.reverse_steps).round().long().tolist()
        assert len(set(times))==len(times)
        for index,timestep in enumerate(times):
            previous=times[index+1] if index+1<len(times) else -1
            # 默认参数捕获当前时间步；重计算不可读取已变更的循环闭包变量。
            def predict(value,current=timestep):
                self.forward_noise_calls+=1
                return self.unet(value.to(self.unet.dtype),current,
                    encoder_hidden_states=self.components.text_embeddings).sample.float()
            if self.config.gradient_checkpointing and torch.is_grad_enabled() and latent.requires_grad:
                epsilon=checkpoint(predict,latent,use_reentrant=False,preserve_rng_state=True)
            else:
                epsilon=predict(latent)
            self._require_finite(epsilon,'噪声网络输出')
            alpha_previous=self.alphas[previous] if previous>=0 else latent.new_tensor(1.0)
            latent=ddim_step(latent,epsilon,self.alphas[timestep],alpha_previous)
            self._require_finite(latent,'逆扩散潜变量')
        result=self._decode(latent,image.shape)
        self._require_finite(result,'扩散净化图片')
        return result
