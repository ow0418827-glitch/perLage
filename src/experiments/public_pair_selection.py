"""固定公开目标的数据集提取、图片标注、分组精筛和统一冻结入口。"""
import argparse
from dataclasses import asdict
import os
from pathlib import Path
import re

from PIL import Image,ImageDraw,ImageFont

from src.paths import PROJECT,DATA,OUTPUT,CONFIGS,PRIVATE,SHARED_MEMORY
from src.vision.audit import digest,freeze_json,read_json,save_json
from src.vision.image_utils import sha256_file
from src.vision.pair_selection import SelectionConfig,LocalSelectionBackend
from src.vision.catalog_training_guard import catalog_training_guard
from src.data_sources.webshop_public import (OWNERSHIP_POLICY,extract_public_candidates,
    semantic_shortlist,download_shortlist)
from src.vision.public_annotations import (ANNOTATION_POLICY,annotate_downloads,contact_sheets,apply_reviews)
from src.experiments.pipeline import select_pair
from src.vision.pair_assets import publish_pair


def validate_config(config):
    if set(config)!={'schema_version','seed','lexical_limit','candidate_limit','text_model','groups'} or config['schema_version']!=1:
        raise ValueError('公开候选准备配置字段或版本无效')
    if any(type(config[k]) is not int or config[k]<1 for k in ('lexical_limit','candidate_limit')):
        raise ValueError('公开候选数量必须为正整数')
    if type(config['seed']) is not int or not config['groups']:
        raise ValueError('固定划分种子或公开目标集合无效')
    for group in config['groups']:
        if (set(group)!={'key','target_id','title_pattern'}
                or not re.fullmatch(r'[A-Za-z0-9_]+',group['key'])
                or not re.fullmatch(r'[A-Za-z0-9]{10}',group['target_id'])):
            raise ValueError('任务标识或公开目标编号无效')
        re.compile(group['title_pattern'])
    if len({g['key'] for g in config['groups']})!=len(config['groups']) or len({g['target_id'] for g in config['groups']})!=len(config['groups']):
        raise ValueError('公开目标及任务标识必须唯一')
    return config


def freeze_plan(config,root):
    freeze_json(Path(root)/'PUBLIC_PLAN.json',dict(schema_version=1,config=config,
        selection=asdict(SelectionConfig()),ownership_policy=OWNERSHIP_POLICY,
        ownership_is_simulated=True,target_ids_fixed=[g['target_id'] for g in config['groups']],
        formal_queries_used=False,victim_memory_used=False,shopping_allowed=False))


def pipeline_config(key,catalog,root):
    """各图对分别冻结，两种方法和全部图对始终指向同一个共享库。"""
    return dict(schema_version=1,method='PerLAGE',public_catalog=str(Path(catalog).resolve()),
        run_dir=str((Path(root)/'groups'/key).resolve()),selection={},
        selection_models=dict(text_model='sentence-transformers/all-MiniLM-L6-v2'),
        generation=dict(lucid=dict(steps=1000,epsilon_levels=16),PerLAGE={}),
        evaluation=dict(shared_pool=str(SHARED_MEMORY),query_manifest=str(PRIVATE/'query_images'/(key+'.json'))))


class SharedBackend:
    def __init__(self,backend):self.backend=backend;self.snapshot=backend.snapshot
    def features(self,record):return self.backend.features(record)
    def recolor_cost(self,a,b):return self.backend.recolor_cost(a,b)
    def close(self):pass


def select_groups(config,root,public,*,device='cuda',backend_factory=LocalSelectionBackend):
    root,public=Path(root),Path(public)
    shortlisted=read_json(public/'semantic_shortlist.json')
    downloaded=read_json(public/'download_manifest.json')
    annotations=read_json(public/'annotation_manifest.json')
    reviews=read_json(public/'visual_reviews.json')
    catalogs=apply_reviews(shortlisted,downloaded,annotations,reviews,public)
    backend=backend_factory(dict(text_model=config['text_model']),public/'selection_features',device)
    rows=[]
    try:
        for group in config['groups']:
            key=group['key'];catalog=catalogs['catalogs'][key]
            definition=pipeline_config(key,catalog,root)
            definition['selection_models']['text_model']=config['text_model']
            cfg_path=public/'pipeline_configs'/(key+'.json');freeze_json(cfg_path,definition)
            group_root=root/'groups'/key
            try:
                selected=select_pair(definition,group_root,device=device,
                    backend_factory=lambda *a,**kw:SharedBackend(backend))
            except ValueError:
                saved=group_root/'selection/SELECTED_PAIR.json'
                if not saved.exists() or read_json(saved).get('status')!='no_eligible_pair':raise
                selected=read_json(saved)
            record=dict(key=key,status=selected['status'],target=selected['target'],
                        selection_path=str((group_root/'selection/SELECTED_PAIR.json').resolve()),
                        selection_sha256=sha256_file(group_root/'selection/SELECTED_PAIR.json'),
                        pipeline_config=str(cfg_path.resolve()),pipeline_config_sha256=sha256_file(cfg_path),
                        candidate_count=len(selected['candidates']),eligible_count=sum(r['eligible'] for r in selected['candidates']))
            if selected['status']=='selected':
                record['source']=selected['source']
                record['score']=next(r for r in selected['candidates'] if r['source_id']==selected['selected_source_id'])
            rows.append(record)
            print(f"{key}：精筛完成，合格 {record['eligible_count']}/{record['candidate_count']}，状态 {record['status']}",flush=True)
    finally:backend.close()
    source_ids=[r['source']['id'] for r in rows if r['status']=='selected']
    if len(source_ids)!=len(set(source_ids)):
        raise ValueError('不同目标选中了重复源商品，需要另立公开去重分派协议，不能静默换次优项')
    target_ids={g['target_id'] for g in config['groups']}
    if target_ids&set(source_ids):raise ValueError('源商品与固定目标集合重叠')
    result=dict(schema_version=1,status='selected' if all(r['status']=='selected' for r in rows) else 'incomplete',
        selection_scope='highest_public_score_within_frozen_shortlists',
        public_plan_sha256=sha256_file(root/'PUBLIC_PLAN.json'),
        public_extraction_sha256=sha256_file(public/'public_extraction.json'),
        semantic_shortlist_sha256=sha256_file(public/'semantic_shortlist.json'),
        download_manifest_sha256=sha256_file(public/'download_manifest.json'),
        annotation_manifest_sha256=sha256_file(public/'annotation_manifest.json'),
        visual_reviews_sha256=sha256_file(public/'visual_reviews.json'),
        groups=rows,shared_memory_scope='shared',shared_memory_read=False,formal_queries_read=False,
        shopping_allowed=False,generation_runs=0,ownership_is_simulated=True,
        thresholds_changed_after_selection=False,private_ranking_used=False)
    freeze_json(root/'PAIR_SELECTION_BUNDLE.json',result)
    return result


def report(root,public):
    root,public=Path(root),Path(public)
    bundle=read_json(root/'PAIR_SELECTION_BUNDLE.json')
    for name,expected in [('PUBLIC_PLAN.json',bundle['public_plan_sha256'])]:
        if sha256_file(root/name)!=expected:raise ValueError('公开选取计划改变')
    for name,field in [('public_extraction.json','public_extraction_sha256'),('semantic_shortlist.json','semantic_shortlist_sha256'),
                       ('download_manifest.json','download_manifest_sha256'),('annotation_manifest.json','annotation_manifest_sha256'),
                       ('visual_reviews.json','visual_reviews_sha256')]:
        if sha256_file(public/name)!=bundle[field]:raise ValueError('公开数据或标注证据改变')
    extracted=read_json(public/'public_extraction.json');shortlisted=read_json(public/'semantic_shortlist.json')
    lines=['# 固定第二组与第九组目标的公开原商品选取','',
           '保留原理发椅和儿童硅胶牙刷目标；从预先划分的模拟可控商品中重新选原商品。',
           '本次只做公开选图，没有生成对抗图、读取共享记忆、运行正式检索或购物。','',
           f"共只读扫描 {extracted['scanned_products']} 条原始商品，合法公开字段记录 {extracted['usable_public_products']} 条。",
           '只投影商品编号、公开名称和第一张主图。未使用数据集类目树、任务指令、价格、购买历史或私有名次。','',
           '模拟可控划分使用固定种子四十二及商品编号散列，目标集合全部排除；这不是实际商家所有权证明。',
           f"每组在至多 {read_json(root/'PUBLIC_PLAN.json')['config']['lexical_limit']} 个固定词法候选中做公开标题语义粗筛，按原有高斯窗口排序，固定取 {read_json(root/'PUBLIC_PLAN.json')['config']['candidate_limit']} 个准备图片。",
           '最终结果表示本次冻结且通过公开质量检查的候选范围内最高分，不是全数据集或真实攻击效果的全局最优。','',
           '| 任务 | 目标编号 | 所选原商品编号 | 图片精筛候选数 | 门槛合格数 | 综合代理分数 |',
           '|---|---|---|---:|---:|---:|']
    for row in bundle['groups']:
        if sha256_file(row['selection_path'])!=row['selection_sha256']:raise ValueError('图对冻结记录改变')
        selected=read_json(row['selection_path'])
        for product in [row['target']]+([row['source']] if row['status']=='selected' else []):
            if sha256_file(product['image'])!=product['image_sha256']:raise ValueError('冻结公开原图改变')
            if product.get('foreground_mask') and sha256_file(product['foreground_mask'])!=product.get('foreground_mask_sha256'):
                raise ValueError('冻结公开前景标注改变')
        eligible=sorted((r for r in selected['candidates'] if r['eligible']),key=lambda r:(-r['score'],r['source_id']))
        if row['status']=='selected' and (not eligible or eligible[0]['source_id']!=row['source']['id']):
            raise ValueError('冻结图对不是公开评分最高的合格候选')
        if row['status']=='selected' and row['score']!=eligible[0]:
            raise ValueError('图对汇总分项与原始候选记录不一致')
        if row['candidate_count']!=len(selected['candidates']) or row['eligible_count']!=len(eligible):
            raise ValueError('图对候选数量或门槛合格数量不一致')
        source=row.get('source',{}).get('id','未选出')
        score=f"{row['score']['score']:.6f}" if 'score' in row else '—'
        lines.append(f"| {row['key']} | {row['target']['id']} | {source} | {row['candidate_count']} | {row['eligible_count']} | {score} |")
    lines+=['','## 公开标题及分项记录','']
    for row in bundle['groups']:
        lines += [f"### {row['key']}",'',f"目标公开名称原文：{row['target']['title']}",'']
        if row['status']=='selected':
            s=row['score'];lines += [f"所选原商品公开名称原文：{row['source']['title']}",'',
                f"原始形态相似度 {s['raw']['shape']:.6f}；公开标题余弦 {s['raw']['semantic_cosine']:.6f}；跨模态差距 {s['raw']['cross_modal_gap']:.6f}；固定试改色感知代价 {s['raw']['pilot_recolor_lpips']:.6f}。",'']
        lines += ['全部候选分项与拒绝原因保存于该任务的选取记录。','']
    lines += ['## 核对与限制','',
        '分割和文字识别使用本地公开模型，自动标注后保留了逐图视觉核对记录。',
        '未降低原有筛选门槛，也未根据私有名次更换图片。公开粗筛、下载、标注、视觉质量拒绝和图片去重的全部记录均保存。',
        '图对与方法无关，两种方法都复用相同公开图对。配置仍指向唯一共享池；本次没有打开该池。',
        '代理分数不是攻击成功概率。后续完整生成和正式模糊查询评估须另行运行。','']
    path=root/'REPORT.md';path.write_text('\n'.join(lines),encoding='utf-8')
    selected=[r for r in bundle['groups'] if r['status']=='selected']
    if selected:
        font_path=Path('C:/Windows/Fonts/msyh.ttc')
        font=ImageFont.truetype(str(font_path),20) if font_path.is_file() else ImageFont.load_default()
        canvas=Image.new('RGB',(1000,len(selected)*610),'white');draw=ImageDraw.Draw(canvas)
        for row_idx,row in enumerate(selected):
            for col,(label,product) in enumerate([('所选原商品',row['source']),('固定目标商品',row['target'])]):
                image=Image.open(product['image']).convert('RGB');image.thumbnail((470,510))
                x=col*500+(500-image.width)//2;y=row_idx*610+75
                display_key={'02_chair':'第二组','09_toothbrush':'第九组'}.get(row['key'],row['key'])
                canvas.paste(image,(x,y));draw.text((col*500+15,row_idx*610+10),f"{display_key} {label}",fill='black',font=font)
                draw.text((col*500+15,row_idx*610+40),product['id'],fill='black',font=font)
        canvas.save(root/'SELECTED_PAIRS.jpg',quality=94)
    return path


def publish_groups(root):
    """将本轮已选图对发布到便于查看和供两种方法读取的分组目录。"""
    root = Path(root)
    bundle = read_json(root / 'PAIR_SELECTION_BUNDLE.json')
    if bundle['status'] != 'selected':
        raise ValueError('全部任务选出合格图对后才能发布正式分组输入')
    records = []
    for row in bundle['groups']:
        if sha256_file(row['selection_path']) != row['selection_sha256']:
            raise ValueError('原公开选取记录改变')
        if not re.fullmatch(r'[A-Za-z0-9_]+', row['key']):
            raise ValueError('分组目录名无效')
        group = DATA / 'public' / 'groups' / row['key']
        selected = publish_pair(row['selection_path'], group, DATA / 'public')
        if any(selected[role]['id'] != row[role]['id'] for role in ('source', 'target')):
            raise ValueError('正式图对与本轮冻结汇总不一致')
        if sha256_file(row['pipeline_config']) != row['pipeline_config_sha256']:
            raise ValueError('原公开流程配置改变')
        definition = read_json(row['pipeline_config'])
        definition.pop('public_catalog')
        definition['public_pair'] = str((group / 'pair.json').resolve())
        definition['run_dir'] = str((OUTPUT / 'runs' / row['key']).resolve())
        definition['selection'] = selected['configuration']
        definition['selection_models'] = selected['binding'].get('models_configuration', {})
        config_path = group / 'pipeline.json'
        freeze_json(config_path, definition)
        records.append(dict(key=row['key'], directory=str(group.resolve()),
            source=selected['source']['image'], target=selected['target']['image'],
            manifest=selected['pair_manifest'], manifest_sha256=selected['pair_manifest_sha256'],
            pipeline_config=str(config_path.resolve()), pipeline_config_sha256=sha256_file(config_path),
            original_selection_sha256=row['selection_sha256']))
    result = dict(schema_version=1, status='published',
                  selected_bundle_sha256=sha256_file(root / 'PAIR_SELECTION_BUNDLE.json'),
                  groups=records, generation_runs=0, shared_memory_read=False)
    freeze_json(root / 'NAMED_PAIRS.json', result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','annotate','select','publish','report','all'),nargs='?',default='all')
    parser.add_argument('--config',type=Path,default=CONFIGS/'selection_groups29.example.json')
    parser.add_argument('--products',type=Path)
    parser.add_argument('--root',type=Path,default=OUTPUT/'public_pair_selection/groups29_20261104')
    parser.add_argument('--public-root',type=Path,default=DATA/'public/selection_runs/groups29_20261104')
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    args=parser.parse_args(argv)
    config=validate_config(read_json(args.config));root=args.root.resolve();public=args.public_root.resolve()
    if not root.is_relative_to(OUTPUT.resolve()) or not public.is_relative_to((DATA/'public').resolve()):
        parser.error('输出及公开资产须位于当前项目自己的目录')
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1')
    freeze_plan(config,root)
    try:
        with catalog_training_guard(allowed_output_roots=[root]) as audit:
            if args.stage in ('prepare','all'):
                if args.products is None:parser.error('公开准备阶段需要 --products 指定原始商品文件')
                extracted=extract_public_candidates(args.products,config['groups'],public,seed=config['seed'],lexical_limit=config['lexical_limit'])
                short=semantic_shortlist(extracted,public,limit=config['candidate_limit'],model_name=config['text_model'],device=args.device)
                download_shortlist(short,public)
            if args.stage in ('annotate','all'):
                freeze_json(root/'ANNOTATION_PLAN.json',dict(policy=ANNOTATION_POLICY,
                    positive_point_from_public_pixels=True,remove_border_connected_background=True,
                    background_color_tolerance=12,actual_visual_review_required=True,
                    source_image_index=0,private_evaluation_used=False))
                annotations=annotate_downloads(read_json(public/'download_manifest.json'),public)
                contact_sheets(read_json(public/'semantic_shortlist.json'),annotations,root/'public_review')
            if args.stage in ('select','all'):
                if not (public/'visual_reviews.json').is_file():
                    save_json(root/'RUN_STATUS.json',dict(status='awaiting_public_visual_review',
                        reason='须依据实际公开图片核对分割和文字区域后记录逐图审核，不能自动伪造审核通过'))
                    print('公开图片标注已准备，等待实际视觉核对记录。');return 2
                selected=select_groups(config,root,public,device=args.device)
            if args.stage in ('select','report','all'):report(root,public)
            if args.stage in ('select','publish','all'):
                if read_json(root/'PAIR_SELECTION_BUNDLE.json')['status']=='selected':publish_groups(root)
                elif args.stage=='publish':raise ValueError('本轮尚未确定全部图对')
            save_json(root/('ACCESS_AUDIT_'+args.stage+'.json'),{k:v for k,v in audit.items() if k!='active'})
        state=('completed' if read_json(root/'PAIR_SELECTION_BUNDLE.json')['status']=='selected' else 'no_eligible_pair') if args.stage in ('select','publish','report','all') else args.stage
        save_json(root/'RUN_STATUS.json',dict(status=state,
                                             victim_memory_read=False,formal_queries_read=False,shopping_runs=0))
        if state=='no_eligible_pair':return 3
    except Exception as error:
        save_json(root/'RUN_STATUS.json',dict(status='failed',stage=args.stage,type=type(error).__name__,reason=str(error)))
        raise
    return 0


if __name__=='__main__':raise SystemExit(main())
