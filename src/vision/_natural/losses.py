"""源项目最终生成细化采用的同一损失函数。"""
import torch
import torch.nn.functional as F
from src.vision.robust_objectives import softmin

def training_public_value(scores,threshold,temperature):
    # 普通条件门槛逐变换对应，净化条件门槛逐请求广播；不能混合不同防御。
    if threshold.shape not in (scores.shape,scores.shape[-1:]):
        raise ValueError('公开训练门槛与逐防御/请求布局不一致')
    gaps=scores-threshold
    if not torch.isfinite(gaps).all():raise FloatingPointError('公开训练余量非有限数')
    return .5*gaps.mean()+.5*softmin(gaps.flatten(),temperature)


def owned_retention_loss(features,teacher_features,joint,config):
    """以自身旧图为参考，不包含其他私有候选或正式排名门槛。"""
    scores=joint.scores(features);reference=joint.scores(teacher_features).detach()
    score_loss=config.temperature*F.softplus((reference+config.desired_owned_score_gain-scores)/config.temperature).mean()
    key=F.normalize(F.normalize(features.float(),dim=-1)+joint.title_feature,dim=-1)
    teacher_key=F.normalize(F.normalize(teacher_features.float(),dim=-1)+joint.title_feature,dim=-1)
    feature_loss=.5*(1-F.cosine_similarity(features.float(),teacher_features.float())).mean()
    feature_loss+=.5*(1-F.cosine_similarity(key,teacher_key)).mean()
    return score_loss+config.feature_weight*feature_loss
