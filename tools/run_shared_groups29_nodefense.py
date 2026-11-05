"""按用户最新范围复用两张完整冻结图，只运行无防御检索与最终购物。"""
import copy
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from src.paths import OUTPUT
from src.vision.audit import read_json,save_json,freeze_json
from src.vision.image_utils import sha256_file
from src.experiments.pipeline import load_frozen

PREVIOUS=OUTPUT/'shared_groups29_20261104_final'
ROOT=OUTPUT/'shared_groups29_20261105_nodefense'


def now():return datetime.now(timezone.utc).isoformat()


def prepare():
    previous_state=read_json(PREVIOUS/'RUN_STATUS.json')
    if previous_state.get('status')!='stopped_by_user_scope_change':
        raise ValueError('原防御队列须先按用户要求停止')
    plan=copy.deepcopy(read_json(PREVIOUS/'BATCH_PLAN.json'))
    for group in plan['groups']:
        frozen=load_frozen(PREVIOUS/'groups'/group['key']/'methods/PerLAGE')
        if frozen['image_sha256']!=group['frozen_image_sha256']:
            raise ValueError('完整冻结图片与共享池计划不一致')
        audit=read_json(PREVIOUS/'groups'/group['key']/'methods/PerLAGE/TRAINING_ACCESS_AUDIT.json')
        if audit['blocked_accesses']:raise ValueError('训练访问审计未通过')
    specs=[s for s in plan['conditions'] if s['label']=='none' and s['kind']=='none']
    if len(specs)!=1:raise ValueError('原计划缺少唯一无防御条件')
    plan['conditions']=specs
    plan['scope_change']=dict(reason='user_requested_undefended_final_tests_only',
        previous_root=str(PREVIOUS),previous_plan_sha256=sha256_file(PREVIOUS/'BATCH_PLAN.json'),
        changed_at=previous_state['stopped_at'],images_retrained=False)
    freeze_json(ROOT/'BATCH_PLAN.json',plan)
    experiment=copy.deepcopy(read_json(PREVIOUS/'EXPERIMENT_PLAN.json'))
    experiment.update(retrieval_conditions=1,scope_change=plan['scope_change'])
    freeze_json(ROOT/'EXPERIMENT_PLAN.json',experiment)
    for name in ('BOTH_FROZEN.json','SHOPPING_AUTHORIZATION.json','INDEPENDENT_POOL_CONTENT_CHECK.json'):
        freeze_json(ROOT/name,read_json(PREVIOUS/name))
    freeze_json(ROOT/'SCOPE_CHANGE.json',dict(plan['scope_change'],
        retrieval_conditions=['none'],shopping_conditions=['none'],
        expected_retrieval_rows=4,expected_rankings=80,expected_shopping_episodes=80,
        existing_defense_results_preserved=True))


def run():
    if (ROOT/'PAUSED_BY_USER.json').exists():
        raise RuntimeError('用户已暂停本轮所有实验，须在用户明确要求继续后解除暂停标记')
    prepare()
    for stage in ('evaluate','shopping','report'):
        completed=ROOT/'stages'/(stage+'.json')
        if completed.exists() and read_json(completed).get('status')=='completed':continue
        log=ROOT/'logs'/(stage+'.log');log.parent.mkdir(parents=True,exist_ok=True)
        actual_stage='verify' if stage=='evaluate' and (ROOT/'retrieval/evaluation.json').exists() else stage
        state=dict(status='running',stage=stage,command_stage=actual_stage,started_at=now(),supervisor_pid=os.getpid(),log=str(log))
        save_json(ROOT/'RUN_STATUS.json',state)
        (ROOT/'RUN_STATUS.md').write_text(f'当前阶段：{stage}\n状态：执行中\n范围：仅无防御检索与最终购物\n日志：{log}\n',encoding='utf-8')
        command=[sys.executable,'-X','utf8','-B','-u','-m','src.experiments.shared_batch',actual_stage,'--root',str(ROOT)]
        started=time.perf_counter()
        with log.open('a',encoding='utf-8') as stream:
            child=subprocess.Popen(command,cwd=PROJECT,stdout=stream,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            state['child_pid']=child.pid;save_json(ROOT/'RUN_STATUS.json',state)
            code=child.wait()
        result=dict(status='completed' if code==0 else 'failed',stage=stage,command_stage=actual_stage,exit_code=code,
            completed_at=now(),seconds=time.perf_counter()-started,log=str(log))
        save_json(completed,result)
        if code:
            save_json(ROOT/'RUN_STATUS.json',result)
            (ROOT/'RUN_STATUS.md').write_text(f'阶段：{stage}\n状态：失败并停止\n日志：{log}\n',encoding='utf-8')
            raise RuntimeError('无防御最终测试阶段失败：'+stage)
    save_json(ROOT/'RUN_STATUS.json',dict(status='completed',completed_at=now(),report=str(ROOT/'REPORT.md')))
    (ROOT/'RUN_STATUS.md').write_text(f'状态：无防御最终测试全部完成\n报告：{ROOT / "REPORT.md"}\n',encoding='utf-8')


if __name__=='__main__':run()
