"""仅执行用户授权的临时放宽洗发水队列；不启动其他商品组。"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import coordinator as common
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[2]


def run():
    scope=common.read(ROOT/'TEMPORARY_SCOPE.json')
    if scope['groups']!=['07_shampoo']:raise ValueError('本次只允许洗发水')
    ready=ROOT/'runs/dominant/07_shampoo/generation/initialization/ready.json'
    import hashlib
    if hashlib.sha256(ready.read_bytes()).hexdigest()!=scope['initial_ready_sha256']:raise ValueError('本轮已完成初始化改变')
    lock=ROOT/'QUEUE.lock'
    if lock.exists():raise RuntimeError('已有监督进程，请先核对现有运行锁')
    common.save(lock,dict(pid=os.getpid(),started_at=common.now(),scope='07_shampoo_only_temporary_relaxation'))
    commands=[
        ('temporary_relaxed_07_shampoo_finish','dominant','finish','worker_temporary_relaxed.py', ['--arm','dominant','--group','07_shampoo','--phase','finish']),
        ('original_07_shampoo_initialize','original','initialize','worker_fileio_recovery.py',['--arm','original','--group','07_shampoo','--phase','initialize']),
        ('original_07_shampoo_finish','original','finish','worker_fileio_recovery.py',['--arm','original','--group','07_shampoo','--phase','finish']),
        ('temporary_shampoo_final_comparison',None,None,'compare_shampoo_temporary.py',[]),
        ('temporary_shampoo_independent_verification',None,None,'compare_shampoo_temporary.py',['--verify'])]
    try:
        for stage,arm,phase,script,arguments in commands:
            completed=ROOT/'stages'/(stage+'.json')
            if completed.exists() and common.read(completed).get('status')=='completed':continue
            log=ROOT/'logs'/(stage+'.log');log.parent.mkdir(parents=True,exist_ok=True)
            started=time.perf_counter()
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',OMP_NUM_THREADS='4')
            with log.open('a',encoding='utf-8') as stream:
                child=subprocess.Popen([sys.executable,'-X','utf8','-B','-u',str(ROOT/script),*arguments],cwd=PROJECT,
                    env=env,stdout=stream,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                while child.poll() is None:
                    common.status(dict(status='running',stage=stage,scope='07_shampoo_only_temporary_relaxation',
                        temporary_halo_multiplier=2.0,default_halo_multiplier=0.5,
                        supervisor_pid=os.getpid(),child_pid=child.pid,log=str(log),progress=common.progress(arm,'07_shampoo',phase)))
                    time.sleep(5)
                code=child.returncode
            record=dict(status='completed' if code==0 else 'failed',stage=stage,exit_code=code,
                        elapsed_seconds=time.perf_counter()-started,completed_at=common.now(),log=str(log))
            common.save(completed,record)
            if code:
                common.status(record)
                raise RuntimeError('临时洗发水队列失败并停止：'+stage)
        common.status(dict(status='completed',stage='shampoo_only_temporary_relaxed_comparison_completed',
                           scope='07_shampoo_only_temporary_relaxation',report=str(ROOT/'REPORT.md'),
                           progress='洗发水完整图、相似度对照及独立复核已完成；主程序默认标准不变。'))
    finally:
        if lock.exists() and common.read(lock).get('pid')==os.getpid():lock.unlink()


if __name__=='__main__':run()
