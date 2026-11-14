"""可恢复的无人值守完整生成队列；全部冻结后自动比较并独立复核。"""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[2]


def now():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path,x):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    p=path.with_suffix(path.suffix+'.tmp');p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');p.replace(path)


def status(data):
    save(ROOT/'RUN_STATUS.json',dict(updated_at=now(),**data))
    text=f"公子，完整流程当前状态：{data['status']}\n\n阶段：{data.get('stage','')}\n\n进度：{data.get('progress','等待进程更新')}\n\n结果报告：{ROOT/'REPORT.md'}\n"
    (ROOT/'RUN_STATUS.md').write_text(text,encoding='utf-8')


def progress(arm,group,phase):
    if not arm:return '全部完整图冻结后的相似度计算或独立复核'
    root=ROOT/'runs'/arm/group/'generation'
    files=([root/'latent_generation_refinement/progress.json'] if phase=='finish' else [])
    files += [root/'initialization/attack/progress.json',root/'initialization/generation/progress.json']
    for path in files:
        if path.exists():
            try:
                d=read(path)
                if 'step' in d:return f"最终细化 {d['step']}/60；状态 {d.get('status')}"
                if 'phase' in d:return f"初始化 {d['phase']}，已完成 {d.get('phase_step')} 步"
                return f"指令生成 {d.get('completed_steps')}/{d.get('total_steps')}"
            except (ValueError,OSError):pass
    return '加载模型、分割或准备颜色阶段'


def run():
    lock=ROOT/'QUEUE.lock'
    if lock.exists():
        prior=read(lock)
        # 只能手动在确认旧监督进程退出后清理过期锁，避免并发占用显存。
        raise RuntimeError('本轮队列已有运行锁：'+str(prior))
    save(lock,dict(pid=os.getpid(),started_at=now()))
    try:
        plan=read(ROOT/'PLAN.json')
        tasks=[]
        for arm,group in plan['execution_order']:
            for phase in ['initialize','finish']:
                stage=f'{arm}_{group}_{phase}'
                cmd=[sys.executable,'-X','utf8','-B','-u',str(ROOT/'worker.py'),'--arm',arm,'--group',group,'--phase',phase]
                tasks.append((stage,arm,group,phase,cmd))
        tasks += [('compare_all_final_images',None,None,None,[sys.executable,'-X','utf8','-B','-u',str(ROOT/'compare_final.py')]),
                  ('independent_verification',None,None,None,[sys.executable,'-X','utf8','-B','-u',str(ROOT/'compare_final.py'),'--verify'])]
        for stage,arm,group,phase,cmd in tasks:
            done=ROOT/'stages'/(stage+'.json')
            if done.exists() and read(done).get('status')=='completed':continue
            log=ROOT/'logs'/(stage+'.log');log.parent.mkdir(parents=True,exist_ok=True)
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',OMP_NUM_THREADS='4')
            started=time.perf_counter()
            with log.open('a',encoding='utf-8') as stream:
                child=subprocess.Popen(cmd,cwd=PROJECT,env=env,stdout=stream,stderr=subprocess.STDOUT,
                                       creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                while child.poll() is None:
                    status(dict(status='running',stage=stage,supervisor_pid=os.getpid(),child_pid=child.pid,
                                log=str(log),progress=progress(arm,group,phase)))
                    time.sleep(5)
                code=child.returncode
            result=dict(status='completed' if code==0 else 'failed',stage=stage,exit_code=code,
                        elapsed_seconds=time.perf_counter()-started,completed_at=now(),log=str(log))
            save(done,result)
            if code:
                status(result)
                raise RuntimeError('完整流程阶段失败并停止：'+stage)
        status(dict(status='completed',stage='full_generation_and_independent_comparison_completed',
                    report=str(ROOT/'REPORT.md'),progress='全部完整图已生成、冻结并复核；请读取最终报告。'))
    finally:
        if lock.exists() and read(lock).get('pid')==os.getpid():lock.unlink()


if __name__=='__main__':run()
