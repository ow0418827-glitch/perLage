"""轻量公开选图：公开索引→粗类型候选→灰度门槛抽样→仅选中图对标注→发布。"""
import argparse
from dataclasses import asdict
import os
from pathlib import Path
import re

from src.paths import DATA,OUTPUT,CONFIGS,PRIVATE,SHARED_MEMORY,PROJECT
from src.vision.audit import digest,freeze_json,read_json,save_json
from src.vision.image_utils import sha256_file
from src.vision.pair_selection import SelectionConfig,LocalSelectionBackend,choose_pair,validate_product,verify_selection
from src.vision.shape_calibration import calibrate,load_calibration
from src.vision.catalog_training_guard import catalog_training_guard
from src.data_sources.webshop_public import extract_public_candidates,random_shortlist,download_shortlist,OWNERSHIP_POLICY
from src.vision.pair_assets import publish_pair


def public_path(value):
    path=Path(value);path=(path if path.is_absolute() else PROJECT/path).resolve()
    if not path.is_relative_to((DATA/'public').resolve()):raise ValueError('选图输入、索引和校准证书须位于当前项目公开目录')
    return path


def validate_config(config):
    required={'schema_version','seed','candidate_limit','calibration_path','groups'}
    if not isinstance(config,dict) or not required<=set(config) or set(config)-required-{'models','type_reviews'} or config['schema_version']!=2:
        raise ValueError('轻量选图须使用第二版配置；旧多指标配置只保留历史读取')
    if type(config['seed']) is not int or type(config['candidate_limit']) is not int or not 1<=config['candidate_limit']<=256:
        raise ValueError('种子须为整数，公开图片候选预算须在一至二百五十六之间')
    if not isinstance(config['calibration_path'],str) or not config['calibration_path'].strip():raise ValueError('须配置公开开发集校准证书路径')
    if not config['groups']:raise ValueError('至少指定一个公开目标')
    for group in config['groups']:
        if (set(group)!={'key','target_id'} or not re.fullmatch(r'[A-Za-z0-9_]+',group['key'])
                or not re.fullmatch(r'[A-Za-z0-9]{10}',group['target_id'])):raise ValueError('组名或固定公开目标编号无效')
    if len({g['key'] for g in config['groups']})!=len(config['groups']) or len({g['target_id'] for g in config['groups']})!=len(config['groups']):
        raise ValueError('组名及公开目标编号须唯一')
    if set(config.get('models',{}))-{'dino_model','batch_size'}:raise ValueError('新选图阶段只使用单一冻结视觉模型')
    return config


def selection_settings(config):
    return SelectionConfig(seed=config['seed'],calibration_path=str(public_path(config['calibration_path'])))


def freeze_plan(config,root):
    from src.vision.product_types import TYPE_RULES_SHA256
    path=public_path(config['calibration_path']);proof=read_json(path)
    load_calibration(path,proof['models'],public_root=DATA/'public')
    freeze_json(Path(root)/'PUBLIC_PLAN.json',dict(schema_version=2,config=config,
        selection=asdict(selection_settings(config)),calibration_sha256=sha256_file(path),
        type_rules_sha256=TYPE_RULES_SHA256,
        implementation_sha256={name:sha256_file(PROJECT/name) for name in ('src/vision/pair_selection.py','src/vision/shape_calibration.py',
            'src/data_sources/webshop_public.py','src/experiments/public_pair_selection.py')},
        ownership_policy=OWNERSHIP_POLICY,target_ids_fixed=[g['target_id'] for g in config['groups']],
        formal_queries_used=False,victim_memory_used=False,shopping_allowed=False,
        annotation_scope='drawn_source_and_target_only',empty_set_policy='stop_without_relaxation'))


def pipeline_config(key,catalog,root,*,selection=None,models=None):
    root=Path(root)
    return dict(schema_version=1,method='PerLAGE',public_catalog=str(Path(catalog).resolve()),
        run_dir=str((root/'groups'/key).resolve()),selection=selection or {},selection_models=models or {},
        generation=dict(lucid=dict(steps=1000,epsilon_levels=16),PerLAGE={}),
        evaluation=dict(shared_pool=str(SHARED_MEMORY),query_manifest=str(PRIVATE/'query_images'/root.name/(key+'.json'))))


def downloaded_product(product,downloads,*,source=False):
    record=dict(id=product['id'],title=product['title'],source_url=product['source_url'],image=downloads['records'][product['id']]['image'])
    if 'type_review' in product:record['type_review']=product['type_review']
    if source:record.update(owned=True,source_note='固定散列规则预先登记的实验模拟可控商品；不声明真实商家所有权')
    return validate_product(record,Path(record['image']).parent,owned=source,generation_ready=False,public_root=DATA/'public')


def select_groups(config,root,public,*,device='cuda',backend_factory=LocalSelectionBackend):
    root,public=Path(root),Path(public)
    short=read_json(public/'candidate_shortlist.json');downloads=read_json(public/'download_manifest.json')
    if downloads['shortlist_sha256']!=digest(short):raise ValueError('图片下载与公开候选集合不一致')
    settings=selection_settings(config);backend=backend_factory(config.get('models',{}),public/'selection_features',device)
    rows=[];used=set();targets={g['target_id'] for g in config['groups']}
    try:
        for group in short['groups']:
            target=downloaded_product(group['target'],downloads)
            candidates=[downloaded_product(p,downloads,source=True) for p in group['candidates'] if p['id'] in downloads['records']]
            catalog=public/'candidate_catalogs'/(group['key']+'.json')
            freeze_json(catalog,dict(schema_version=2,target=target,candidates=candidates))
            if not candidates:raise ValueError('本组所有公开候选下载失败：'+group['key'])
            result=choose_pair(target,candidates,backend,settings,excluded_source_ids=used|targets,public_root=DATA/'public')
            result.update(target=target,binding=dict(public_catalog_sha256=sha256_file(catalog),models_configuration=config.get('models',{}),
                calibration_sha256=sha256_file(settings.calibration_path),public_plan_sha256=sha256_file(root/'PUBLIC_PLAN.json')),
                download_failures={p['id']:downloads['failures'][p['id']] for p in group['candidates'] if p['id'] in downloads['failures']})
            if result['selected_source_id']:
                result['source']=next(p for p in candidates if p['id']==result['selected_source_id']);used.add(result['selected_source_id'])
            path=root/'groups'/group['key']/'selection/DRAWN_PAIR.json';freeze_json(path,result)
            rows.append(dict(key=group['key'],status=result['status'],selection_path=str(path.resolve()),selection_sha256=sha256_file(path),
                target=target,source=result.get('source'),candidate_count=len(candidates),eligible_count=sum(r['eligible'] for r in result['candidates'])))
    finally:backend.close()
    bundle=dict(schema_version=2,status='selected' if all(r['status']=='selected' for r in rows) else 'no_eligible_pair',
        selection_scope='uniform_draw_after_public_shape_and_type_gates_v2',groups=rows,
        public_plan_sha256=sha256_file(root/'PUBLIC_PLAN.json'),shared_memory_read=False,formal_queries_read=False,generation_runs=0)
    freeze_json(root/'PAIR_SELECTION_BUNDLE.json',bundle);return bundle


def selected_inputs(root,public):
    bundle=read_json(Path(root)/'PAIR_SELECTION_BUNDLE.json')
    if bundle['status']!='selected':raise ValueError('存在空合格集合；停止本轮，不放宽门槛或重新抽样')
    short=read_json(Path(public)/'candidate_shortlist.json');downloads=read_json(Path(public)/'download_manifest.json')
    groups=[];ids=set()
    for row in bundle['groups']:
        if sha256_file(row['selection_path'])!=row['selection_sha256']:raise ValueError('抽样证据改变')
        selection=read_json(row['selection_path']);verify_selection(selection)
        original=next(g for g in short['groups'] if g['key']==row['key'])
        source=next(p for p in original['candidates'] if p['id']==selection['selected_source_id'])
        groups.append(dict(key=row['key'],target=original['target'],candidates=[source]));ids.update((source['id'],original['target']['id']))
    selected_short=dict(schema_version=2,groups=groups,original_shortlist_sha256=digest(short))
    subset=dict(records={pid:downloads['records'][pid] for pid in sorted(ids)},original_downloads_sha256=digest(downloads))
    return selected_short,subset


def annotate_selected(root,public):
    from src.vision.public_annotations import annotate_downloads,contact_sheets
    root,public=Path(root),Path(public);short,downloads=selected_inputs(root,public)
    freeze_json(public/'selected_shortlist.json',short)
    annotations=annotate_downloads(downloads,public/'selected_annotations')
    contact_sheets(short,annotations,root/'public_review')
    return annotations


def publish_groups(root,public):
    from src.vision.public_annotations import apply_reviews
    root,public=Path(root),Path(public);short,downloads=selected_inputs(root,public)
    annotations=read_json(public/'selected_annotations/annotation_manifest.json')
    reviews=read_json(public/'selected_visual_reviews.json')
    catalogs=apply_reviews(short,downloads,annotations,reviews,public/'selected_annotations')
    bundle=read_json(root/'PAIR_SELECTION_BUNDLE.json');records=[]
    for row in bundle['groups']:
        original=read_json(row['selection_path']);document=read_json(catalogs['catalogs'][row['key']])
        selected=dict(original)
        for role,product in [('target',document['target']),('source',document['candidates'][0])]:
            for field in ('id','title','image'):
                if product[field]!=original[role][field]:raise ValueError('选中图对标注改变了公开身份或原图')
            if 'type_review' in original[role]:product['type_review']=original[role]['type_review']
            selected[role]=validate_product(product,Path(product['image']).parent,owned=role=='source',public_root=DATA/'public')
        selected['annotation_binding']=dict(drawn_pair_sha256=row['selection_sha256'],annotations_sha256=digest(annotations),reviews_sha256=digest(reviews))
        selected_path=root/'groups'/row['key']/'selection/SELECTED_PAIR.json';freeze_json(selected_path,selected)
        # 协议版本目录保护此前已冻结的分组 source/target 及其实验结果。
        directory=DATA/'public/groups'/root.name/row['key']
        published=publish_pair(selected_path,directory,DATA/'public')
        cfg=pipeline_config(row['key'],catalogs['catalogs'][row['key']],root,
            selection=published['configuration'],models=published['binding']['models_configuration'])
        cfg.pop('public_catalog');cfg['public_pair']=published['pair_manifest']
        cfg['run_dir']=str((OUTPUT/'runs'/root.name/row['key']).resolve())
        config_path=directory/'pipeline.json';freeze_json(config_path,cfg)
        records.append(dict(key=row['key'],directory=str(directory.resolve()),source=published['source']['image'],target=published['target']['image'],
            manifest=published['pair_manifest'],pipeline_config=str(config_path.resolve())))
    result=dict(schema_version=2,status='published',groups=records,bundle_sha256=sha256_file(root/'PAIR_SELECTION_BUNDLE.json'),
        generation_runs=0,shared_memory_read=False)
    freeze_json(root/'NAMED_PAIRS.json',result);return result


def report(root,public):
    root=Path(root);bundle=read_json(root/'PAIR_SELECTION_BUNDLE.json')
    if bundle['public_plan_sha256']!=sha256_file(root/'PUBLIC_PLAN.json'):raise ValueError('公开报告与冻结计划不一致')
    lines=['# 轻量公开图对抽样','',
        '仅以类型兼容、灰度全局特征门槛及去重判断合格，使用冻结种子在合格集合均匀抽样。没有按最高相似分选图，也没有读取正式记忆或查询。','',
        '|组别|目标编号|抽中原商品|图片候选数|合格数|单个合格候选抽中概率|','|---|---|---|---:|---:|---:|']
    for row in bundle['groups']:
        if sha256_file(row['selection_path'])!=row['selection_sha256']:raise ValueError('公开抽样证据改变')
        selection=read_json(row['selection_path'])
        if row['status']=='selected':verify_selection(selection)
        source=row['source']['id'] if row['source'] else '未选出';n=row['eligible_count']
        probability=f'1/{n}' if n else '无合格集合'
        lines.append(f"|{row['key']}|{row['target']['id']}|{source}|{row['candidate_count']}|{n}|{probability}|")
    lines+=['','选中图对才进行主体与印刷区标注；逐图视觉核对后发布到协议版本目录，每组继续使用 source、target 文件名。',
        '外形分数是代理，不是攻击成功概率；此协议不保证原图在未知记忆库中位于前三之外。开发集商品与正式候选、目标不重叠。',
        '主体、品牌文字等公开核对不合格时停止发布，不根据正式结果重选。正式生成及购物不属于本次选图命令。','']
    path=root/'REPORT.md';path.write_text('\n'.join(lines),encoding='utf-8');return path


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('calibrate','prepare','select','annotate','publish','report','all'),nargs='?',default='all')
    parser.add_argument('--config',type=Path,default=CONFIGS/'selection_groups29.example.json')
    parser.add_argument('--products',type=Path);parser.add_argument('--development',type=Path)
    parser.add_argument('--index',type=Path,default=DATA/'public/public_products_v2.sqlite')
    parser.add_argument('--verify-source',action='store_true')
    parser.add_argument('--root',type=Path,default=OUTPUT/'public_pair_selection/shape_v2')
    parser.add_argument('--public-root',type=Path,default=DATA/'public/selection_runs/simple_v2')
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    args=parser.parse_args(argv);config=validate_config(read_json(args.config));root=args.root.resolve();public=public_path(args.public_root)
    if not root.is_relative_to(OUTPUT.resolve()):parser.error('运行输出须位于当前项目自己的输出目录')
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1')
    try:
        with catalog_training_guard(allowed_output_roots=[root]) as audit:
            if args.stage=='calibrate':
                if args.development is None:parser.error('校准需要 --development 指定实际公开视觉标签清单')
                backend=LocalSelectionBackend(config.get('models',{}),public/'calibration_features',args.device)
                try:calibrate(public_path(args.development),public_path(config['calibration_path']),backend,public_root=DATA/'public')
                finally:backend.close()
                save_json(root/'ACCESS_AUDIT_calibrate.json',{k:v for k,v in audit.items() if k!='active'})
                print('公开开发集门槛已拟合并冻结。');return 0
            if args.stage!='report':freeze_plan(config,root)
            if args.stage in ('prepare','all'):
                if args.products is None:parser.error('准备阶段需要 --products 指定原始商品文件')
                proof=read_json(public_path(config['calibration_path']))
                reviews=read_json(public_path(config['type_reviews'])) if config.get('type_reviews') else {}
                extracted=extract_public_candidates(args.products,config['groups'],public,seed=config['seed'],
                    lexical_limit=config['candidate_limit'],index_path=public_path(args.index),excluded_ids=proof['development_product_ids'],
                    type_reviews=reviews,verify_source=args.verify_source)
                download_shortlist(random_shortlist(extracted,public),public)
            if args.stage in ('select','all'):
                selected=select_groups(config,root,public,device=args.device)
                if selected['status']!='selected':
                    report(root,public)
                    save_json(root/'RUN_STATUS.json',dict(status='no_eligible_pair',generation_runs=0,threshold_relaxed=False))
                    save_json(root/('ACCESS_AUDIT_'+args.stage+'.json'),{k:v for k,v in audit.items() if k!='active'})
                    return 3
            if args.stage in ('annotate','all'):annotate_selected(root,public)
            if args.stage in ('publish','all'):
                if not (public/'selected_visual_reviews.json').is_file():
                    save_json(root/'RUN_STATUS.json',dict(status='awaiting_selected_public_visual_review',selected_only=True,generation_runs=0))
                    print('仅选中图对已准备标注；等待实际逐图视觉核对后发布。');return 2
                publish_groups(root,public)
            if args.stage in ('select','publish','report','all'):report(root,public)
            save_json(root/('ACCESS_AUDIT_'+args.stage+'.json'),{k:v for k,v in audit.items() if k!='active'})
        if args.stage=='report':return 0
        state={'prepare':'prepared','select':'drawn_awaiting_selected_annotations','annotate':'awaiting_selected_public_visual_review',
            'publish':'published','all':'published','report':'reported'}[args.stage]
        save_json(root/'RUN_STATUS.json',dict(status=state,victim_memory_read=False,formal_queries_read=False,shopping_runs=0))
        return 0
    except Exception as error:
        save_json(root/'RUN_STATUS.json',dict(status='failed',stage=args.stage,type=type(error).__name__,reason=str(error)))
        raise


if __name__=='__main__':raise SystemExit(main())
