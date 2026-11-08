"""公开筛选和训练的文件边界；限制用于复现审计，并非操作系统沙箱。"""
from contextlib import contextmanager
from src.paths import ProjectPath as Path
import os
import sys
from src.paths import PROJECT, PRIVATE, OUTPUT, SHARED_MEMORY


@contextmanager
def catalog_training_guard(*, allowed_output_roots, private_paths=()):
    allowed = tuple(Path(p).resolve() for p in allowed_output_roots)
    private = (PRIVATE.resolve(), SHARED_MEMORY.parent.parent.resolve(),
               (PROJECT / 'src/experiments/formal_queries.py').resolve()) + tuple(Path(p).resolve() for p in private_paths)
    forbidden_names = {'queries.json', 'evaluation.json', 'results.json', 'test_results.json'}
    state = {'active': True, 'checked_accesses': 0, 'blocked_accesses': []}

    def audit(event, args):
        if not state['active'] or event not in ('open', 'os.listdir', 'os.scandir') or not args:
            return
        value = args[0]
        if not isinstance(value, (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(value)).resolve()
        # 核心代码、运行时和公开权重可读；共享库和评估文件始终不可读。
        is_private = (any(path == p or path.is_relative_to(p) for p in private)
                      or path.name.lower().startswith(('memory', 'history', 'shared_memory', 'clean_memory', 'adversarial_memory'))
                      and path.suffix.lower() in ('.json', '.jsonl')
                      or path.name.lower() in forbidden_names)
        is_other_run = path.is_relative_to(OUTPUT.resolve()) and not any(path.is_relative_to(p) for p in allowed)
        if is_private or is_other_run:
            state['blocked_accesses'].append(str(path))
            raise PermissionError('公开筛选及训练禁止访问私有记忆或评估结果：' + str(path))
        state['checked_accesses'] += 1

    sys.addaudithook(audit)
    try:
        yield state
    finally:
        state['active'] = False
