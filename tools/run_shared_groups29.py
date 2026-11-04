"""两组公开成图、共享记忆复用及最终测试的可恢复串行编排。"""
import argparse
import copy
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.paths import DATA, OUTPUT, CONFIGS
from src.vision.audit import read_json, save_json, freeze_json
from src.vision.image_utils import sha256_file
from src.memory.shared_pool import build_shared_pool

ROOT = OUTPUT / 'shared_groups29_20261104'
GROUPS = ('02_chair', '09_toothbrush')
OLD_POOL = PROJECT.parent / 'lage/src/output/lage_runs/catalog_shared_memory_20261103/shared_memory_pool.json'
BASE = DATA / 'private/reused_shared_20261104'
ACTIVE = DATA / 'private/shared_memory.json'


def now():return datetime.now(timezone.utc).isoformat()


def prepare():
    BASE.mkdir(parents=True, exist_ok=True)
    original = BASE / 'original_shared_pool.json'
    expected = sha256_file(OLD_POOL)
    if original.exists():
        if sha256_file(original) != expected:raise ValueError('原共享记忆快照改变')
    else:shutil.copyfile(OLD_POOL, original)
    source = read_json(original)
    if len(source) != 100:raise ValueError('旧共享池数量与预期不一致')
    normalized = []
    for row in source:
        value = copy.deepcopy(row)
        real = value.get('Completed') is True and value.get('RecordOrigin') != 'simulated_catalog_gallery'
        value['RecordKind'] = 'purchase_memory' if real else 'simulated_gallery'
        if not real:value['Completed'] = False
        normalized.append(value)
    normalized_path = BASE / 'normalized_import.json'
    freeze_json(normalized_path, normalized)
    baseline = BASE / 'baseline_shared_memory.json'
    if not baseline.exists():build_shared_pool([normalized_path], baseline, kind='preserve')
    freeze_json(BASE / 'IMPORT.json', dict(source=str(OLD_POOL), source_sha256=expected,
        baseline=str(baseline), baseline_sha256=sha256_file(baseline), records=len(source),
        purchase_memories=sum(r['RecordKind']=='purchase_memory' for r in normalized),
        simulated_references=sum(r['RecordKind']=='simulated_gallery' for r in normalized),
        source_project_read_only=True))
    source_config = PROJECT.parent / 'lage/src/configs/llm.local.json'
    destination = CONFIGS / 'llm.local.json'
    if not destination.exists():shutil.copyfile(source_config, destination)
    for key in GROUPS:
        config = read_json(DATA / 'public/groups' / key / 'pipeline.json')
        config['run_dir'] = str((ROOT / 'groups' / key).resolve())
        config['evaluation']['shared_pool'] = str(ACTIVE.resolve())
        config['evaluation']['query_manifest'] = str((DATA / 'private/query_images' / (key+'.json')).resolve())
        freeze_json(ROOT / 'configs' / (key+'.json'), config)
    freeze_json(ROOT / 'EXPERIMENT_PLAN.json', dict(schema_version=1, groups=list(GROUPS),
        baseline_pool=str(baseline), baseline_sha256=sha256_file(baseline), active_pool=str(ACTIVE),
        training_uses_private_memory=False, both_images_frozen_before_evaluation=True,
        simultaneous_two_image_substitution=True, retrieval_conditions=26, requests_per_group=20,
        shopping_authorized=True, shopping_conditions=['none'], shopping_top_k=3,
        stop_on_failure=True, change_seeds_or_retrain_from_private_results=False))
    print('共享池已独立复制：100 条；本轮配置已冻结。', flush=True)


def run():
    prepare()
    phases = [('generate_'+key, [sys.executable,'-X','utf8','-B','-u','-m',
        'src.experiments.pipeline','generate','--method','PerLAGE','--config',str(ROOT/'configs'/(key+'.json'))]) for key in GROUPS]
    phases += [(stage, [sys.executable,'-X','utf8','-B','-u','-m','src.experiments.shared_batch',
                       stage,'--root',str(ROOT)]) for stage in ('finalize','evaluate','shopping','report')]
    for stage, command in phases:
        completed = ROOT / 'stages' / (stage+'.json')
        if completed.exists() and read_json(completed).get('status')=='completed':continue
        log = ROOT / 'logs' / (stage+'.log')
        log.parent.mkdir(parents=True, exist_ok=True)
        state=dict(status='running',stage=stage,started_at=now(),supervisor_pid=__import__('os').getpid(),log=str(log))
        save_json(ROOT/'RUN_STATUS.json',state)
        (ROOT/'RUN_STATUS.md').write_text(f'当前阶段：{stage}\n状态：执行中\n日志：{log}\n',encoding='utf-8')
        started=time.perf_counter()
        with log.open('a',encoding='utf-8') as stream:
            child=subprocess.Popen(command,cwd=PROJECT,stdout=stream,stderr=subprocess.STDOUT,
                                   creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            state['child_pid']=child.pid;save_json(ROOT/'RUN_STATUS.json',state)
            code=child.wait()
        result=dict(status='completed' if code==0 else 'failed',stage=stage,exit_code=code,
                    completed_at=now(),seconds=time.perf_counter()-started,log=str(log))
        save_json(completed,result)
        if code:
            save_json(ROOT/'RUN_STATUS.json',result)
            (ROOT/'RUN_STATUS.md').write_text(f'阶段：{stage}\n状态：失败并停止\n日志：{log}\n',encoding='utf-8')
            raise RuntimeError('实验阶段失败：'+stage)
    save_json(ROOT/'RUN_STATUS.json',dict(status='completed',completed_at=now(),report=str(ROOT/'REPORT.md')))
    (ROOT/'RUN_STATUS.md').write_text(f'状态：全部完成\n报告：{ROOT / "REPORT.md"}\n',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','all'))
    args=parser.parse_args()
    prepare() if args.stage=='prepare' else run()
