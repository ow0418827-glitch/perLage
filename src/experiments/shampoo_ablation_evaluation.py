"""全部成图冻结后，使用同一一百八十九条全局库评估并独立复算。"""
import copy
import gc
import hashlib
import json
import math
import os
import statistics

import numpy as np
from PIL import Image

from src.paths import ProjectPath as Path, PROJECT, SHARED_MEMORY
from src.visual.audit import read_json, freeze_json, save_json
from src.visual.image_utils import sha256_file
from src.agent.memory.balanced_pool import validate_balanced_pool

LABELS = {
    'clean': '未修改原图',
    'lucid_original_16': '原始像素方法旧图（十六级）',
    'lucid_4': '原始像素方法（四级）',
    'lucid_8': '原始像素方法（八级）',
    'perlage_original_no_shadow': '感知方法历史图（无影子池）',
    'perlage_pixel_control': '感知方法含像素（共享起点配对对照）',
    'perlage_latent_only': '感知方法仅潜变量（匹配原对照设置）',
    'perlage_legacy_reference': '更早感知方法完整图（附加参考）',
    'perlage_color_revision_reference': '后续主色方案完整图（附加参考）',
}


class MemoizedEncoder:
    """只复用同一进程内相同图像和文字的结果；独立复算重建整个实例。"""
    def __init__(self, encoder):
        self.encoder = encoder
        self.snapshot = encoder.snapshot
        self.device = encoder.device
        self.supports_image_weight = encoder.supports_image_weight
        self.values = {}
        self.last_encoding_audit = None

    def encode(self, path):
        key = ('image', sha256_file(path))
        if key not in self.values:
            self.values[key] = (self.encoder.encode(path), None)
        return self.values[key][0].copy()

    def encode_joint(self, text, path, image_weight=0.5):
        key = ('joint', text, sha256_file(path), image_weight)
        if key not in self.values:
            value = self.encoder.encode_joint(text,path,image_weight=image_weight)
            self.values[key] = (value, copy.deepcopy(self.encoder.last_encoding_audit))
        value,audit = self.values[key]
        self.last_encoding_audit = copy.deepcopy(audit)
        return value.copy()

    def close(self):
        self.encoder.close()
        self.values.clear()


def freeze_inputs(root):
    from src.visual.attack_completion import validate_attack_completion
    from src.visual._perlage.evidence import validate_complete
    from src.experiments.formal_queries import VAGUE_INSTRUCTIONS
    root=Path(root)
    historical=read_json(root/'HISTORICAL_IMAGES.json')
    original_pair=read_json(root/'provenance/lucid_original.json')['settings']
    expected_pair={role+'_image_sha256':original_pair[role+'_sha256'] for role in ('source','target')}
    for role in ('source','target'):
        if sha256_file(root/'data/07_shampoo'/(role+'.jpg'))!=expected_pair[role+'_image_sha256']:
            raise ValueError('评估图对与原始冻结对照不一致')
    variants={name:dict(row) for name,row in historical.items()}
    for level in (4,8):
        folder=root/f'generation/lucid_{level}'
        metadata=read_json(folder/'completion.json')
        validate_attack_completion(metadata,1000,expected_pair)
        if abs(metadata['attack']['epsilon']-level/255)>1e-12:
            raise ValueError('像素预算与本轮固定设置不一致')
        variants[f'lucid_{level}']=dict(image=metadata['output_image'],sha256=metadata['output_image_sha256'])
    perlage=read_json(root/'generation/perlage_latent_only/completion.json')
    validate_complete(perlage['perlage_ready'],perlage['output_image'])
    if any(perlage[role+'_image_sha256']!=value for role,value in (
            ('source',expected_pair['source_image_sha256']),('target',expected_pair['target_image_sha256']))):
        raise ValueError('感知消融没有绑定同一公开图对')
    definition=read_json(root/'data/07_shampoo/latent_only_matched.json')
    definition_sha=hashlib.sha256(json.dumps(definition,sort_keys=True).encode('utf-8')).hexdigest()
    if perlage['perlage_ready']['configuration_sha256']!=definition_sha:
        raise ValueError('感知消融冻结后其配置被修改')
    if perlage['perlage_ready']['auxiliary_max_levels']!=0:
        raise ValueError('仅潜变量成图含像素残差')
    variants['perlage_latent_only']=dict(image=perlage['output_image'],sha256=perlage['output_image_sha256'])
    paired=read_json(root/'generation/perlage_pixel_control/completion.json')
    validate_complete(paired['perlage_ready'],paired['output_image'])
    paired_definition=read_json(root/'data/07_shampoo/pixel_control_matched.json')
    paired_definition_sha=hashlib.sha256(json.dumps(paired_definition,sort_keys=True).encode('utf-8')).hexdigest()
    if paired['perlage_ready']['configuration_sha256']!=paired_definition_sha:
        raise ValueError('含像素配对对照冻结后其配置被修改')
    normalized=copy.deepcopy(paired_definition)
    normalized['pixel_optimization']=False
    normalized['initialization']['optimizer'].update(pixel_clean_steps=0,pixel_diffusion_steps=0,pixel_levels=0)
    normalized['latent']['auxiliary_levels']=0
    if normalized!=definition:
        raise ValueError('感知配对对照除了像素步骤与残差之外还改变了配置')
    if any(paired[role+'_image_sha256']!=expected_pair[role+'_image_sha256'] for role in ('source','target')):
        raise ValueError('含像素配对对照没有使用同一图对')
    parent_init=read_json(root/'generation/perlage_latent_only/initialization/ready.json')
    paired_init=read_json(root/'generation/perlage_pixel_control/initialization/ready.json')
    if parent_init['candidates']['generation_only']['sha256']!=paired_init['candidates']['generation_only']['sha256']:
        raise ValueError('感知配对对照的公共潜变量起点不同')
    paired_steps=read_json(root/'generation/perlage_pixel_control/initialization/attack/metadata.json')
    for phase,count in [('pixel_clean',20),('pixel_diffusion',60)]:
        if max(r['step'] for r in paired_steps['history'] if r['phase']==phase)!=count:
            raise ValueError('含像素配对对照没有完整完成像素阶段')
    variants['perlage_pixel_control']=dict(image=paired['output_image'],sha256=paired['output_image_sha256'])
    source=root/'data/07_shampoo/source.jpg'
    variants['clean']=dict(image=str(source),sha256=sha256_file(source))
    variants={name:variants[name] for name in LABELS if name in variants}
    for item in variants.values():
        if sha256_file(item['image'])!=item['sha256']:
            raise ValueError('冻结图片内容改变')
    bank=validate_balanced_pool(SHARED_MEMORY)
    sources=[r for r in bank if r['FinalASIN']=='B08XYBS678' and r['EntryType']=='error']
    if len(sources)!=1 or sources[0]['PrimaryImage']['SHA256']!=sha256_file(source):
        raise ValueError('共享池未唯一绑定本轮源商品')
    query=root/'evaluation_inputs/query.png'
    plan=dict(schema_version=1,protocol='shampoo_single_global_189_controlled_image_ablation_v1',
        pool_path=str(SHARED_MEMORY),pool_sha256=sha256_file(SHARED_MEMORY),pool_count=len(bank),
        source_memory_id=sources[0]['Id'],source_asin='B08XYBS678',target_asin='B00BDJODWI',
        source_image=str(source),target_image=str(root/'data/07_shampoo/target.jpg'),
        source_image_sha256=expected_pair['source_image_sha256'],target_image_sha256=expected_pair['target_image_sha256'],
        public_definition_sha256=sha256_file(root/'data/07_shampoo/definition.json'),
        latent_definition_sha256=sha256_file(root/'data/07_shampoo/latent_only_matched.json'),
        pixel_control_definition_sha256=sha256_file(root/'data/07_shampoo/pixel_control_matched.json'),
        paired_protocol_amendment_sha256=sha256_file(root/'data/07_shampoo/PAIRED_PROTOCOL_AMENDMENT.json'),
        query_image=str(query),query_image_sha256=sha256_file(query),
        queries=[dict(task_id=f'vague_{i:02d}',instruction=q) for i,q in enumerate(VAGUE_INSTRUCTIONS,1)],
        variants=variants,condition='none',key_image_weight=0.5,query_image_weight=0.5,index_text='title',
        full_global_bank_used_for_every_query=True,formal_queries_used_during_training=False,
        source_title_changed=False,only_one_memory_image_changed_per_arm=True,shopping_runs=0,
        primary_perlage_control='perlage_pixel_control',common_latent_prefix_exactly_shared=True,
        secondary_references=['perlage_legacy_reference','perlage_color_revision_reference'])
    freeze_json(root/'evaluation/PLAN.json',plan)
    return plan


def _inputs(root):
    root=Path(root);plan=read_json(root/'evaluation/PLAN.json')
    if sha256_file(plan['pool_path'])!=plan['pool_sha256']:
        raise ValueError('基础共享池已改变')
    bank=validate_balanced_pool(plan['pool_path'])
    if sha256_file(plan['query_image'])!=plan['query_image_sha256']:
        raise ValueError('正式模糊参考图已改变')
    for role in ('source','target'):
        if sha256_file(plan[role+'_image'])!=plan[role+'_image_sha256']:
            raise ValueError('公开图对在冻结后已改变')
    for name,key in [('definition.json','public_definition_sha256'),('latent_only_matched.json','latent_definition_sha256'),
                     ('pixel_control_matched.json','pixel_control_definition_sha256'),
                     ('PAIRED_PROTOCOL_AMENDMENT.json','paired_protocol_amendment_sha256')]:
        if sha256_file(root/'data/07_shampoo'/name)!=plan[key]:
            raise ValueError('公开生成配置在冻结后已改变')
    for row in plan['variants'].values():
        if sha256_file(row['image'])!=row['sha256']:
            raise ValueError('待比较的冻结成图已改变')
    return root,plan,bank


def compute_rankings(root):
    from src.experiments.shared_evaluation import make_eval_encoder
    from src.visual.visual_retrieval import VisualMemoryRetriever,resolve_image
    root,plan,base=_inputs(root)
    encoder=MemoizedEncoder(make_eval_encoder('cuda'))
    source_id=plan['source_memory_id'];rows=[]
    try:
        for name,item in plan['variants'].items():
            bank=copy.deepcopy(base)
            for entry in bank:
                entry['PrimaryImage']['LocalPath']=str(resolve_image(entry,plan['pool_path']))
            original=copy.deepcopy(bank)
            if name!='clean':
                entry=next(e for e in bank if e['Id']==source_id)
                entry['RetrievalImage']=dict(ASIN=plan['source_asin'],LocalPath=item['image'],SHA256=item['sha256'])
            if [{k:v for k,v in e.items() if k!='RetrievalImage'} for e in bank]!=original:
                raise ValueError('评估副本意外改变历史、标题或其他条目')
            bank_path=root/'evaluation/banks'/(name+'.json')
            freeze_json(bank_path,bank)
            ranker=VisualMemoryRetriever(encoder,bank_path,cache_dir=None)
            queries=[]
            for query in plan['queries']:
                if sha256_file(plan['query_image'])!=plan['query_image_sha256']:
                    raise ValueError('正式参考图在逐请求检索时改变')
                ranked=ranker.rank(bank,query['instruction'],plan['query_image'])
                if len(ranked)!=189:
                    raise ValueError('没有检索完整的一百八十九条全局库')
                source=next(r for r in ranked if r['Id']==source_id)
                queries.append(dict(task_id=query['task_id'],source_rank=source['rank'],source_score=source['score'],
                    ranking=[dict(Id=r['Id'],rank=r['rank'],score=r['score'],image_sha256=r['image_sha256']) for r in ranked]))
            row=dict(variant=name,queries=queries,encoder=copy.deepcopy(encoder.snapshot),
                     encoder_device=str(encoder.device),bank_sha256=sha256_file(bank_path))
            rows.append(row)
            print(f'{LABELS[name]}：平均名次 {statistics.mean(q["source_rank"] for q in queries):.2f}',flush=True)
    finally:
        encoder.close()
        gc.collect()
    return rows


def compute_similarities(root):
    from skimage.metrics import structural_similarity,peak_signal_noise_ratio
    from src.experiments.shared_evaluation import make_eval_encoder
    from src.visual.encoders.clip import make_encoder
    import torch
    root,plan,_=_inputs(root)
    with Image.open(plan['source_image']) as image:
        original=np.asarray(image.convert('RGB')).copy()
    rows=[]
    for name,item in plan['variants'].items():
        with Image.open(item['image']) as image:
            pixels=np.asarray(image.convert('RGB')).copy()
        if pixels.shape!=original.shape:
            raise ValueError('成图尺寸与源图不一致')
        error=np.abs(pixels.astype(float)-original.astype(float))
        psnr=peak_signal_noise_ratio(original,pixels,data_range=255) if error.max() else None
        rows.append(dict(variant=name,source_ssim=float(structural_similarity(original,pixels,channel_axis=2,data_range=255)),
            source_psnr_db=float(psnr) if psnr is not None else None,
            source_psnr_infinite=not bool(error.max()),
            source_mae_levels=float(error.mean()),source_max_levels=int(error.max()),
            image_sha256=item['sha256']))
    for label,factory in [('training_proxy_2',lambda:make_encoder(surrogate=2,device='cuda')),
                          ('held_out',lambda:make_eval_encoder('cuda'))]:
        encoder=factory()
        try:
            target=encoder.encode(plan['target_image']);source=encoder.encode(plan['source_image'])
            query=encoder.encode(plan['query_image'])
            for row in rows:
                vector=encoder.encode(plan['variants'][row['variant']]['image'])
                row[label]=dict(target_image_cosine=float(vector@target),source_image_cosine=float(vector@source),
                    query_image_cosine=float(vector@query),encoder=encoder.snapshot)
        finally:
            if hasattr(encoder,'close'):
                encoder.close()
            del encoder
            gc.collect();torch.cuda.empty_cache()
    return rows


def evaluate(root):
    import torch
    torch.set_num_threads(4)
    root=Path(root)
    freeze_inputs(root)
    plan_sha=sha256_file(root/'evaluation/PLAN.json')
    rows=compute_rankings(root)
    if sha256_file(root/'evaluation/PLAN.json')!=plan_sha:raise ValueError('评估中冻结计划发生改变')
    freeze_json(root/'evaluation/RANKINGS.json',dict(rows=rows,pool_count=189,plan_sha256=plan_sha))
    similarities=compute_similarities(root)
    if sha256_file(root/'evaluation/PLAN.json')!=plan_sha:raise ValueError('评估中冻结计划发生改变')
    freeze_json(root/'evaluation/SIMILARITIES.json',similarities)


def verify(root):
    import torch
    torch.set_num_threads(4)
    root=Path(root)
    recorded=read_json(root/'evaluation/RANKINGS.json')
    plan_sha=sha256_file(root/'evaluation/PLAN.json')
    if not plan_sha or recorded.get('plan_sha256')!=plan_sha:
        raise ValueError('独立复核的正式请求或冻结计划摘要不符')
    old=recorded['rows']
    actual=compute_rankings(root)
    error=0.;count=0
    if len(old)!=len(actual):raise ValueError('复核条件数量不同')
    for a,b in zip(old,actual):
        if a['variant']!=b['variant'] or a['encoder']!=b['encoder'] or a['bank_sha256']!=b['bank_sha256']:
            raise ValueError('复核模型或实验条件不同')
        if len(a['queries'])!=20 or len(b['queries'])!=20:
            raise ValueError('复核请求须为完整二十条')
        for q,r in zip(a['queries'],b['queries']):
            if q['task_id']!=r['task_id'] or len(q['ranking'])!=len(r['ranking']):
                raise ValueError('复核正式请求不同')
            if q['source_rank']!=r['source_rank'] or not math.isfinite(q['source_score']) or abs(q['source_score']-r['source_score'])>2e-5:
                raise ValueError('源商品名次或分数汇总与独立复核不一致')
            for x,y in zip(q['ranking'],r['ranking']):
                if (x['Id'],x['rank'],x['image_sha256'])!=(y['Id'],y['rank'],y['image_sha256']):
                    raise ValueError('独立复核排名、标识或图片不同')
                delta=abs(x['score']-y['score']);error=max(error,delta);count+=1
                if not math.isfinite(delta) or delta>2e-5:
                    raise ValueError('独立复核分数误差过大')
    old_sim=read_json(root/'evaluation/SIMILARITIES.json');new_sim=compute_similarities(root)
    if len(old_sim)!=len(old) or len(new_sim)!=len(actual):
        raise ValueError('图像相似度复核条件缺失或重复')
    for a,b in zip(old_sim,new_sim):
        if a['variant']!=b['variant'] or a['image_sha256']!=b['image_sha256']:
            raise ValueError('图像相似度复核输入不同')
        for field in ('source_ssim','source_mae_levels','source_max_levels'):
            if abs(a[field]-b[field])>1e-10:raise ValueError('像素质量复核不同')
        if a['source_psnr_infinite']!=b['source_psnr_infinite'] or (
                a['source_psnr_db']!=b['source_psnr_db'] and (
                    a['source_psnr_db'] is None or b['source_psnr_db'] is None or
                    abs(a['source_psnr_db']-b['source_psnr_db'])>1e-10)):
            raise ValueError('源图峰值信噪比复核不同')
        for model in ('training_proxy_2','held_out'):
            if a[model]['encoder']!=b[model]['encoder']:
                raise ValueError('图像相似度编码器身份不同')
            for field in ('target_image_cosine','source_image_cosine','query_image_cosine'):
                delta=abs(a[model][field]-b[model][field]);error=max(error,delta)
                if not math.isfinite(delta) or delta>2e-5:raise ValueError('图像编码复核不同')
    if sha256_file(root/'evaluation/PLAN.json')!=plan_sha:
        raise ValueError('独立复核期间冻结计划发生改变')
    freeze_json(root/'evaluation/INDEPENDENT_VERIFICATION.json',dict(status='passed',plan_sha256=plan_sha,
        fresh_encoder_process=True,encoder_device=actual[0]['encoder_device'],full_bank_scores_recomputed=count,
        variants=len(actual),query_count=20,max_absolute_error=error,
        rankings_sha256=sha256_file(root/'evaluation/RANKINGS.json'),
        similarities_sha256=sha256_file(root/'evaluation/SIMILARITIES.json')))
    report(root)


def report(root):
    root=Path(root);plan=read_json(root/'evaluation/PLAN.json')
    ranks=read_json(root/'evaluation/RANKINGS.json')['rows']
    similarities={r['variant']:r for r in read_json(root/'evaluation/SIMILARITIES.json')}
    verification=read_json(root/'evaluation/INDEPENDENT_VERIFICATION.json')
    if verification['status']!='passed':raise ValueError('报告前须独立复算通过')
    summary=[]
    for item in ranks:
        name=item['variant'];queries=item['queries'];numbers=[q['source_rank'] for q in queries]
        summary.append(dict(variant=name,label=LABELS[name],mean_rank=statistics.mean(numbers),
            minimum_rank=min(numbers),maximum_rank=max(numbers),ranks=numbers,
            mean_score=statistics.mean(q['source_score'] for q in queries),
            top1=sum(r<=1 for r in numbers),top3=sum(r<=3 for r in numbers),top5=sum(r<=5 for r in numbers),
            query_count=len(numbers),similarity=similarities[name]))
    freeze_json(root/'SUMMARY.json',dict(rows=summary,verification=verification,pool_count=189,
        protocol=plan['protocol'],shopping_runs=0))
    lines=['公子，本轮已完成统一共享池下的洗发水图像对照。','',
        '九组各二十条正常记录加一条错误记录，共一百八十九条。每个条件均检索同一完整全局池，只替换洗发水对应的一张记忆图；名称、历史、其他一百八十八条记录和正式请求保持一致。',
        '正常与错误是实验角色；公开商品参考仍标记为未观察到购买，不能将其统计为真实购买历史。','',
        '无防御，商品图片与真实名称各占一半，查询图片与模糊文字各占一半。复用无名称、无品牌的参考图和固定二十种正式措辞。排名越小越靠前。本轮不执行最终购物。','',
        '|图片条件|源图结构相似度|源图平均像素差|源图最大像素差|主代理目标图余弦|留出模型目标图余弦|平均名次|前三 / 二十|前五 / 二十|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in summary:
        s=row['similarity'];lines.append(f'|{row["label"]}|{s["source_ssim"]:.6f}|{s["source_mae_levels"]:.4f}|{s["source_max_levels"]}|{s["training_proxy_2"]["target_image_cosine"]:.6f}|{s["held_out"]["target_image_cosine"]:.6f}|{row["mean_rank"]:.2f}|{row["top3"]}/20|{row["top5"]}/20|')
    lines+=['','原始像素方法四级与八级都执行完整一千步，旧十六级图保持原始字节，仅重新编码与检索。',
        '历史无影子池完整图保留原始字节并重新检索。独立重跑的公共潜变量前缀与历史前缀有数值差异，因此另增加本轮共享起点的含像素配对对照：复用本轮完整三十步指令生成与两段各四十步潜变量初始化的同一冻结前缀，再执行二十步普通像素优化、六十步净化像素优化、颜色协调和完整四十步最终细化。仅潜变量版本与此配对对照的起点图片摘要完全相同，训练措辞、种子、模型和质量门槛一致。',
        '更早的感知图与后续主色方案另列参考，涉及方法版本或颜色规则差异，不用于归因像素步骤的贡献。',
        f'独立进程重新创建编码器，共复算 {verification["full_bank_scores_recomputed"]} 个全库分数，并复核图像相似度。最大分数误差为 {verification["max_absolute_error"]:.8g}。','',
        '结构相似度、平均与最大像素差均相对未修改源图；目标图余弦为视觉编码相似度。源图与目标图尺寸不同，不把二者的像素误差当成外观相似度。检索效果与最终购物结果分别定义，本轮只报告检索。','',
        '逐请求全库排名、模型身份、输入摘要及独立复核记录位于本结果目录的评估子目录。','']
    for name in ('perlage_pixel_control','perlage_latent_only'):
        stage=read_json(root/'generation'/name/'latent_generation_refinement/metadata.json')
        completed=max(item['step'] for item in stage['history'])
        selected=stage['best']['step']
        lines.append(f'{LABELS[name]}：最终潜变量细化实际完成 {completed}/{stage["config"]["steps"]} 步，'
            f'原有生成期选择规则保留第 {selected} 步候选；选择没有使用正式查询或共享池名次。')
    lines+=['含像素配对对照保留第零步，表示完成整个日程后仍保留最终细化的输入候选；不能将它表述为选中了后续四十步中的新候选。仅潜变量图保留第二十步候选。','']
    lookup={r['variant']:r for r in summary}
    comparisons=[]
    for name,reference in [('lucid_4','lucid_original_16'),('lucid_8','lucid_original_16'),
                            ('perlage_latent_only','perlage_pixel_control'),
                            ('perlage_latent_only','perlage_original_no_shadow')]:
        current,baseline=lookup[name],lookup[reference]
        comparisons.append(dict(variant=name,reference=reference,
            source_ssim_delta=current['similarity']['source_ssim']-baseline['similarity']['source_ssim'],
            held_out_target_cosine_delta=current['similarity']['held_out']['target_image_cosine']-baseline['similarity']['held_out']['target_image_cosine'],
            mean_rank_delta=current['mean_rank']-baseline['mean_rank'],top3_delta=current['top3']-baseline['top3']))
    freeze_json(root/'PAIRED_DELTAS.json',comparisons)
    for row in comparisons:
        lines.append(f'{LABELS[row["variant"]]}相对{LABELS[row["reference"]]}：源图结构相似度变化 {row["source_ssim_delta"]:+.6f}；留出模型目标图余弦变化 {row["held_out_target_cosine_delta"]:+.6f}；平均名次变化 {row["mean_rank_delta"]:+.2f}（负值表示提前）；前三命中次数变化 {row["top3_delta"]:+d}/20。')
    if lookup['clean']['top3']==20:
        lines+=['','未修改源图已经在全部二十条请求中进入前三；修改图原本就有的前三召回不能归因于新增攻击效果。']
    lines+=['','二十条请求共用同一参考图，属于措辞变化，不是二十个独立商品样本；本轮结论对应这一固定图对与共享池。','']
    (root/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    _plot(root,plan,summary)


def _plot(root,plan,summary):
    os.environ.setdefault('MPLCONFIGDIR',str(PROJECT/'tmp/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']
    plt.rcParams['axes.unicode_minus']=False
    names=['target','clean','lucid_original_16','lucid_4','lucid_8','perlage_original_no_shadow','perlage_pixel_control','perlage_latent_only']
    lookup={r['variant']:r for r in summary}
    fig,axes=plt.subplots(2,4,figsize=(18,10),layout='constrained')
    for axis,name in zip(axes.flat,names):
        if name=='target':
            with Image.open(plan['target_image']) as image:axis.imshow(image.convert('RGB'))
            axis.set_title('目标商品原图\n仅供视觉参照，正式查询使用无名称参考图',fontsize=10)
            axis.axis('off')
            continue
        with Image.open(plan['variants'][name]['image']) as image:axis.imshow(image.convert('RGB'))
        row=lookup[name];s=row['similarity']
        axis.set_title(f'{LABELS[name]}\n结构相似度 {s["source_ssim"]:.4f}；平均名次 {row["mean_rank"]:.2f}',fontsize=11)
        axis.axis('off')
    fig.suptitle('同一图对、同一一百八十九条共享池、同一二十个模糊请求',fontsize=15)
    fig.savefig(root/'IMAGE_COMPARISON.png',dpi=150);plt.close(fig)
