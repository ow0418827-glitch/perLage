"""仅为本轮进度文件原子替换加入短暂有界重试；方法与质量参数不变。"""
import json
from pathlib import Path
import runpy
import time
ROOT=Path(__file__).resolve().parent
_replace_original=Path.replace


def replace_with_bounded_retry(self,target):
    target_path=Path(target)
    scoped=(self.name.endswith('.json.tmp') and target_path.suffix=='.json'
            and self.resolve().is_relative_to(ROOT))
    if not scoped:return _replace_original(self,target)
    for attempt in range(8):
        try:return _replace_original(self,target)
        except PermissionError as exc:
            if getattr(exc,'winerror',None) not in (5,32) or attempt==7:raise
            delay=.1*(2**min(attempt,3))
            try:
                with (ROOT/'fileio_recovery_events.jsonl').open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps(dict(file=str(target_path),attempt=attempt+1,wait_seconds=delay,winerror=exc.winerror),ensure_ascii=False)+'\n')
            except OSError:pass
            time.sleep(delay)


if __name__=='__main__':
    Path.replace=replace_with_bounded_retry
    runpy.run_path(str(ROOT/'worker.py'),run_name='__main__')
