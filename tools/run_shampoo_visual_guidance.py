"""复用洗发水旧测试图片，在同一共享修改库中比较提交一条和三条记忆。"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime,timezone
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from src.paths import OUTPUT,CONFIGS
from src.vision.audit import read_json,save_json,freeze_json
from src.vision.image_utils import sha256_file
from src.experiments.recall_depth_evaluation import replace_reused_image

OLD=PROJECT.parent/'lage/src/output/lage_runs'
NIGHT=OUTPUT/'shared_groups29_20261105_neutral'
DEFAULT_ROOT=OUTPUT/'shampoo_k13_20261106'
GUIDANCE=('Some recalled records are public catalog references explicitly marked as having no observed purchase. '
    'Treat only completed purchase records as purchase histories. '
    'Compare the reference image with the recalled images before choosing a product.')
SOURCE_ASIN='B08XYBS678'
TARGET_ASIN='B00BDJODWI'


def now():return datetime.now(timezone.utc).isoformat()


def copy_frozen(source,destination):
    source,destination=Path(source),Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    if not destination.exists():shutil.copyfile(source,destination)
    if sha256_file(source)!=sha256_file(destination):raise ValueError('复用素材与原文件不一致')
    return destination


def seed_original_cache(root,bank,extra_roots,forbidden_hashes):
    from src.memory.multimodal_memory import _assert_original_product_image,validated_image_bytes
    from src.experiments.shared_shopping import CATALOG
    from src.environment.disk_catalog import DiskCatalog
    destination=Path(root)/'shopping/observed_images/original_image_cache'
    known={e['FinalASIN']:e['PrimaryImage']['SHA256'] for e in bank}
    imported=[]
    with DiskCatalog.open(CATALOG) as catalog:
        for given in extra_roots:
            given=Path(given)
            for p in sorted((given/'image_records').glob('*.json')):
                record=read_json(p);asin=record['asin'];_assert_original_product_image(record,asin)
                if record['sha256'] in forbidden_hashes:continue
                if asin in known and record['sha256']!=known[asin]:continue
                if catalog.product(asin).get('MainImage')!=record['url']:continue
                original=(given/record['path']).resolve()
                if not original.is_relative_to((given/'assets').resolve()):raise ValueError('原图缓存路径越界')
                validated_image_bytes(dict(LocalPath=str(original),SHA256=record['sha256']))
                name=hashlib.sha256((asin+'\0'+record['url']).encode('utf-8')).hexdigest()+'.json'
                target_record=destination/'image_records'/name
                if target_record.exists():continue
                target=destination/'assets'/(record['sha256']+original.suffix.lower())
                copy_frozen(original,target)
                value=dict(record,path=target.relative_to(destination).as_posix(),role='original_product')
                freeze_json(target_record,value)
                imported.append(dict(asin=asin,url=record['url'],sha256=record['sha256']))
    save_json(Path(root)/'ORIGINAL_CACHE_IMPORT.json',dict(imported=imported,original_images_only=True))


def prepare(root):
    from src.configuration.llm_config import load_model_config
    root=Path(root).resolve()
    old_ready=OLD/'catalog_shampoo_20261028/generated/07_shampoo/ready.json'
    ready=read_json(old_ready)
    if ready['method']!='nf_lage' or ready['status']!='completed':raise ValueError('旧洗发水修改图未完整完成')
    required={'foreground_segmentation','foreground_statistics','color_harmonization','latent_generation_refinement'}
    if set(ready['stages'])!=required:raise ValueError('旧图缺少完整四阶段')
    preserved={str(old_ready):sha256_file(old_ready)}
    for stage in ready['stages'].values():
        if stage['status']!='completed':raise ValueError('旧图阶段尚未完成')
        for p,s in stage['files'].items():
            if sha256_file(p)!=s:raise ValueError('旧图四阶段冻结证据改变')
            preserved[p]=s
    latent=ready['stages']['latent_generation_refinement']
    if latent['steps_completed']!=latent['steps_requested']:raise ValueError('旧图潜变量步骤未完成')
    legacy_bank_path=OLD/'webshop_vground_shampoo_hidden_label_20261104/07_shampoo_bank_none.json'
    old_bank=read_json(legacy_bank_path)
    old_source=next(e for e in old_bank if e.get('FinalASIN')==SOURCE_ASIN)
    image=Path(ready['candidates']['final']['image'])
    if sha256_file(image)!=ready['candidates']['final']['sha256'] or sha256_file(image)!=old_source['RetrievalImage']['SHA256']:
        raise ValueError('旧购物实际使用的图片与完整冻结图不一致')
    query=OLD/'shampoo_hidden_label_20261028/query_without_label.png'
    former=read_json(OLD/'webshop_vground_shampoo_hidden_label_20261104/top3/results.json')[0]['trajectory']
    images=former['PolicyTrace'][0]['InputImages']
    if not any(im.get('Role')=='query' and im['SHA256']==sha256_file(query) for im in images):
        raise ValueError('复用参考图与旧测试实际输入不一致')
    assets=root/'inputs/07_shampoo'
    modified=copy_frozen(image,assets/'modified.png');query_copy=copy_frozen(query,assets/'query.png')
    base_path=NIGHT/'shopping/policy_banks/modified.json';base=read_json(base_path)
    if len(base)!=100:raise ValueError('昨晚共享库须为一百条，禁止沿用旧分组小库')
    source=next(e for e in base if e.get('FinalASIN')==SOURCE_ASIN and e.get('EntryType')=='error')
    source_image=copy_frozen(source['PrimaryImage']['LocalPath'],assets/'source.jpg')
    target_image=copy_frozen(OLD/'catalog_shampoo_no_shadow_20261029/assets/target.jpg',assets/'target.jpg')
    if sha256_file(source_image)!=ready['source_sha256'] or sha256_file(target_image)!=ready['target_sha256']:
        raise ValueError('源图和目标图与旧成图公开输入不同')
    bank_path=root/'shopping/policy_bank_modified.json'
    bank,source_id=replace_reused_image(base,bank_path,SOURCE_ASIN,modified,ready['source_sha256'])
    freeze_json(bank_path,bank)
    old_plan=read_json(NIGHT/'BATCH_PLAN.json')
    queries=[dict(task_id=q['task_id'],instruction=q['instruction'],query_image=str(query_copy),
                  query_image_sha256=sha256_file(query_copy)) for q in old_plan['groups'][0]['queries']]
    if len(queries)!=20:raise ValueError('须沿用昨晚二十种模糊措辞')
    guidance_path=root/'inputs/visual_guidance.json';freeze_json(guidance_path,[dict(note1=GUIDANCE)])
    model=load_model_config(CONFIGS/'llm.local.json').public()
    if model!=read_json(NIGHT/'shopping/COMPLETED.json')['model']:raise ValueError('购物模型与昨晚不同')
    paths=[image,query,old_ready,legacy_bank_path,base_path,modified,query_copy,source_image,target_image,bank_path,guidance_path]
    for p in paths:preserved[str(p)]=sha256_file(p)
    plan=dict(schema_version=1,condition='none',bank_state='modified',recall_counts=[1,3],
        shopping_authorized=True,
        queries=queries,pool_count=100,bank_path=str(bank_path),base_bank_path=str(base_path),
        source_memory_id=source_id,source_asin=SOURCE_ASIN,target_asin=TARGET_ASIN,
        source=dict(id=SOURCE_ASIN,title=source['PrimaryImage']['Title'],image=str(source_image),image_sha256=sha256_file(source_image)),
        modified_image_sha256=sha256_file(modified),query_image_sha256=sha256_file(query_copy),
        guidance_text=GUIDANCE,guidance_path=str(guidance_path),guidance_sha256=sha256_file(guidance_path),
        model=model,input_sha256=preserved,legacy_image_method=ready['method'],legacy_ready_sha256=sha256_file(old_ready),
        legacy_image_new_generation=False,user_authorized_legacy_query_reuse=True,
        query_origin='old_catalog_reference_with_entire_label_removed_not_newly_generated',
        full_global_ranking=True,source_forced_to_rank_one=False,other_99_records_unchanged=True,
        record_kinds={k:sum(e['RecordKind']==k for e in bank) for k in ['purchase_memory','simulated_gallery']},
        settings=dict(temperature=.5,policy_max_tokens=2048,policy_max_retries=3,min_request_interval=12,num_steps=15),
        policy_images='query_and_recalled_images_only_catalog_images_audited_not_sent')
    freeze_json(root/'PLAN.json',plan);freeze_json(root/'PLAN_DIGEST.json',dict(sha256=sha256_file(root/'PLAN.json')))
    freeze_json(root/'SHOPPING_AUTHORIZATION.json',dict(shopping_allowed=True,condition='none',bank='modified',
        recall_counts=[1,3],episodes=40,source='user_requested_reuse_of_shampoo_test_images'))
    freeze_json(root/'ASSET_REVIEW.json',dict(legacy_four_stages_verified=True,
        legacy_evidence_files=sum(len(s['files']) for s in ready['stages'].values()),
        query_actually_viewed=True,no_visible_brand_or_product_name=True,
        reused_exact_image_from_previous_shopping_test=True,private_ranking_used_for_asset_selection=False,
        original_project_read_only=True,new_generation_runs=0))
    cache_roots=[NIGHT/'shopping/observed_images/original_image_cache',
        OLD/'webshop_vground_shampoo_hidden_label_20261104/original_image_cache',
        OLD/'webshop_vground_shampoo_20261104/original_image_cache']
    seed_original_cache(root,bank,cache_roots,{sha256_file(modified),sha256_file(query_copy)})
    return plan


def run(root):
    root=Path(root).resolve();prepare(root)
    if not (root/'STARTED.json').exists():freeze_json(root/'STARTED.json',dict(started_at=now(),supervisor_pid=os.getpid()))
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    for stage in ['rank','verify','shopping','report']:
        done=root/'stages'/(stage+'.json')
        if done.exists() and read_json(done).get('status')=='completed':continue
        log=root/'logs'/(stage+'.log');log.parent.mkdir(parents=True,exist_ok=True)
        state=dict(status='running',stage=stage,started_at=now(),supervisor_pid=os.getpid(),log=str(log))
        save_json(root/'RUN_STATUS.json',state);tick=time.perf_counter()
        with log.open('a',encoding='utf-8') as stream:
            child=subprocess.Popen([sys.executable,'-X','utf8','-B','-u','-m','src.experiments.recall_depth_evaluation',
                stage,'--root',str(root),'--device','cuda'],cwd=PROJECT,env=env,stdout=stream,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            state['child_pid']=child.pid;save_json(root/'RUN_STATUS.json',state);code=child.wait()
        if code:
            save_json(root/'RUN_STATUS.json',dict(status='failed',stage=stage,failed_at=now(),log=str(log)))
            raise RuntimeError('洗发水提交数量对照失败并停止：'+stage)
        save_json(done,dict(status='completed',stage=stage,completed_at=now(),seconds=time.perf_counter()-tick))
    save_json(root/'RUN_STATUS.json',dict(status='completed',completed_at=now(),report=str(root/'REPORT.md')))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args();prepare(args.root) if args.prepare_only else run(args.root)
