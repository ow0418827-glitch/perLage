"""Frozen nine-group feature-text + target-original-image retrieval; no generation or shopping."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'tools')]
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')

from src.visual.audit import digest, freeze_json, read_json, save_json
from src.visual.image_utils import sha256_file
from run_groups9_multimodal_20261110 import MemoizedEncoder, lock, make_ranker

RUN = PROJECT / 'output/groups9_target_image_20261111'
FROZEN_PLAN = PROJECT / 'output/groups9_multimodal_20261110/EXPERIMENT_PLAN.json'
FROZEN_PLAN_SHA = '8821a4e55e26c7edac4c1ddfe8058ae0136bcac9f24c4727fc0c2c890d2010b3'
METHODS = ('lucid', 'perlage_no_pixel')
EXPECTED = 'ccaf4c2a8c6795055c7968d5b0940923af2a464f35a3d2509e31414227d4e569'


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def check(path, expected):
    if sha256_file(path) != expected:
        raise ValueError('冻结输入摘要改变：' + str(path))


def status(run, state, **extra):
    value = dict(status=state, updated_at=now(), pid=os.getpid(), **extra)
    save_json(run / 'RUN_STATUS.json', value)
    (run / 'RUN_STATUS.md').write_text('# 九组特点文字加目标原图检索状态\n\n' +
        '\n'.join(f'- {k}: {v}' for k, v in value.items()) + '\n', encoding='utf-8')


def build_plan(run):
    check(FROZEN_PLAN, FROZEN_PLAN_SHA)
    current = read_json(FROZEN_PLAN)
    if tuple(current['expected_methods']) != METHODS or current['query_mode'] != 'joint':
        raise ValueError('冻结计划方法或查询模式不符')
    plan = dict(
        schema_version=1,
        execution='scheduled_retrieval_only',
        output=str(run),
        frozen_current_plan=dict(path=str(FROZEN_PLAN), sha256=FROZEN_PLAN_SHA),
        query_protocol=dict(
            mode='joint', image_weight=0.5,
            text='exact_20_feature_texts_from_20261110_multimodal_round',
            images='target_original_product_images_per_group',
            image_source_note=('查询图片为每组目标商品原图，含目标商品自身品牌印刷与可读文字；'
                               '这是用户于2026年11月11日明确指定的新测试条件，'
                               '不改变项目默认的独立生成无品牌模糊照片协议'),
        ),
        memory_protocol=dict(count=189, key_image_weight=0.5, index_text='product_feature',
            scope='complete_global_bank_for_every_request', training_reads_allowed=False),
        pool=current['pool'], pool_sha256=current['pool_sha256'], pool_count=current['pool_count'],
        encoder_config=current['encoder_config'], encoder_config_sha256=current['encoder_config_sha256'],
        condition='none',
        expected_rankings=720,
        expected_independent_score_checks=136080,
        shopping_enabled=False,
        generation_enabled=False,
        scope_amendment=dict(
            user_specified=True,
            specified_on='2026-11-11',
            change='查询图片由独立生成无品牌照片改为每组目标商品原图',
            reason='用户要求新增D组条件：特点文字加目标商品图，观察与A/B/C条件的排名差异',
            default_protocol_unchanged=True,
            query_images_contain_brand_print=True,
            shopping_not_included=True,
            retrieval_only=True,
        ),
        methods={},
    )
    for method in METHODS:
        previous_groups = current['methods'][method]['groups']
        groups = []
        for group in previous_groups:
            texts = [(q['task_id'], q['instruction']) for q in group['queries']]
            if len(texts) != 20 or len({t[0] for t in texts}) != 20:
                raise ValueError('特点文字不是二十条唯一请求：' + group['key'])
            queries = [dict(task_id=task_id, instruction=instruction, query_mode='joint',
                query_image=group['target']['image'],
                query_image_sha256=group['target']['image_sha256'])
                for task_id, instruction in texts]
            groups.append(dict(
                key=group['key'], name=group['name'],
                source=group['source'], target=group['target'],
                source_memory_id=group['source_memory_id'],
                frozen_file=group['frozen_file'], frozen_file_sha256=group['frozen_file_sha256'],
                frozen_image=group['frozen_image'], frozen_image_sha256=group['frozen_image_sha256'],
                banks=group['banks'],
                queries=queries,
                training_overlap=group.get('training_overlap'),
            ))
        plan['methods'][method] = dict(groups=groups)
    return plan


def validate(run):
    plan_file = run / 'SCHEDULED_INPUTS.json'
    if not EXPECTED:
        raise ValueError('入口尚未绑定冻结计划摘要')
    check(plan_file, EXPECTED)
    plan = read_json(plan_file)
    if (tuple(plan['methods']) != METHODS or plan['shopping_enabled'] is not False
            or plan['generation_enabled'] is not False or plan['condition'] != 'none'
            or plan['expected_rankings'] != 720 or plan['expected_independent_score_checks'] != 136080
            or plan['query_protocol']['mode'] != 'joint' or plan['query_protocol']['image_weight'] != 0.5
            or plan['memory_protocol']['key_image_weight'] != 0.5
            or plan['query_protocol']['images'] != 'target_original_product_images_per_group'):
        raise ValueError('本轮仅检索范围或协议不符')
    origin = plan['frozen_current_plan']
    check(origin['path'], origin['sha256'])
    if origin['sha256'] != FROZEN_PLAN_SHA:
        raise ValueError('绑定来源计划不是冻结的十一月十日图文计划')
    current = read_json(origin['path'])
    check(plan['encoder_config'], plan['encoder_config_sha256'])
    check(plan['pool'], plan['pool_sha256'])
    checked_images = {}
    for method in METHODS:
        groups = plan['methods'][method]['groups']
        if len(groups) != 9 or [g['key'] for g in groups] != [g['key'] for g in current['methods'][method]['groups']]:
            raise ValueError('九组范围改变')
        for group, previous in zip(groups, current['methods'][method]['groups']):
            for field in ('source', 'target', 'source_memory_id', 'banks', 'frozen_image', 'frozen_image_sha256'):
                if group[field] != previous[field]:
                    raise ValueError('冻结商品、库或图片身份改变：' + field)
            for pathkey, shakey in [('frozen_file', 'frozen_file_sha256'), ('frozen_image', 'frozen_image_sha256')]:
                check(group[pathkey], group[shakey])
            expected_texts = [(q['task_id'], q['instruction']) for q in previous['queries']]
            if [(q['task_id'], q['instruction']) for q in group['queries']] != expected_texts:
                raise ValueError('没有逐字使用十一月十日二十条特点文字')
            for query in group['queries']:
                if (query['query_mode'] != 'joint' or query['query_image'] != group['target']['image']
                        or query['query_image_sha256'] != group['target']['image_sha256']):
                    raise ValueError('查询模式或目标原图绑定改变')
            check(group['target']['image'], group['target']['image_sha256'])
            if group['target']['image'] not in checked_images:
                checked_images[group['target']['image']] = group['target']['image_sha256']
            for name in ('clean', 'modified'):
                bound = group['banks'][name]
                check(bound['path'], bound['sha256'])
                bank = read_json(bound['path'])
                if len(bank) != 189 or len({e['Id'] for e in bank}) != 189:
                    raise ValueError('库不是一百八十九条唯一记忆')
                source = next(e for e in bank if e['Id'] == group['source_memory_id'])
                if source['FinalASIN'] != group['source']['id']:
                    raise ValueError('源记忆商品编号不符')
                for entry in bank:
                    image = entry.get('RetrievalImage') or entry['PrimaryImage']
                    if image['LocalPath'] not in checked_images:
                        check(image['LocalPath'], image['SHA256'])
                        checked_images[image['LocalPath']] = image['SHA256']
    return plan


def output_file(run, method, group, bank):
    return run / 'methods' / method / 'retrieval' / group / (bank + '.json')


def fresh_encoder(plan):
    from src.experiments.shared_evaluation import make_eval_encoder
    raw = make_eval_encoder('cuda', plan['encoder_config'])
    surrogate = raw.snapshot['surrogate']
    if surrogate['device'] != 'cuda' or surrogate['loading']['actual_activation_modules'] != {'QuickGELU': 24, 'GELU': 0}:
        raw.close()
        raise ValueError('实际设备或快速激活实现不符')
    previous = read_json(plan['frozen_current_plan']['path'])
    first = previous['methods']['lucid']['groups'][0]
    old = read_json(PROJECT / 'output/groups9_multimodal_20261110/methods/lucid/retrieval' / first['key'] / 'clean.json')
    if digest(raw.snapshot) != digest(old['protocol']['encoder']):
        save_json(RUN / 'ENCODER_MISMATCH.json', dict(actual=raw.snapshot,
            expected=old['protocol']['encoder'], actual_sha256=digest(raw.snapshot),
            expected_sha256=digest(old['protocol']['encoder'])))
        raw.close()
        raise ValueError('实际编码器快照与上轮冻结配置不一致')
    return raw, MemoizedEncoder(raw)


def result_checks(result, plan, method, group, name):
    if (result['plan_sha256'] != EXPECTED or result['method'] != method or result['group'] != group['key']
            or result['bank'] != name or result['pool_count'] != 189 or len(result['rankings']) != 20):
        raise ValueError('既有结果绑定或范围不符')
    for query, row in zip(group['queries'], result['rankings']):
        if query['task_id'] != row['task_id'] or len(row['ranking']) != 189:
            raise ValueError('结果未覆盖正式请求与全部候选')
        if len({r['Id'] for r in row['ranking']}) != 189:
            raise ValueError('候选身份重复')
        source = next(r for r in row['ranking'] if r['Id'] == group['source_memory_id'])
        if source['rank'] != row['source_rank']:
            raise ValueError('源记忆名次提取不符')
        for index, item in enumerate(row['ranking'], 1):
            if (item['rank'] != index or item['bank_sha256'] != group['banks'][name]['sha256']
                    or item['query_text_sha256'] != hashlib.sha256(query['instruction'].encode('utf-8')).hexdigest()
                    or item['query_image_sha256'] != query['query_image_sha256']):
                raise ValueError('逐候选请求或库身份不符')


def retrieve(run, method):
    plan = validate(run)
    raw, encoder = fresh_encoder(plan)
    try:
        for group in plan['methods'][method]['groups']:
            for name in ('clean', 'modified'):
                path = output_file(run, method, group['key'], name)
                if path.exists():
                    result_checks(read_json(path), plan, method, group, name)
                    continue
                status(run, 'running', stage='retrieve', method=method, group=group['key'], bank=name)
                bank, ranker = make_ranker(encoder, group, name)
                rows = []
                for query in group['queries']:
                    ranking = ranker.rank(bank, query['instruction'], query['query_image'])
                    if len(ranking) != 189:
                        raise ValueError('排名没有覆盖全部共享库')
                    source = next(r for r in ranking if r['Id'] == group['source_memory_id'])
                    rows.append(dict(task_id=query['task_id'], source_rank=source['rank'], source_score=source['score'],
                        selected_ids=[r['Id'] for r in ranking[:3]], ranking=ranking))
                result = dict(method=method, group=group['key'], bank=name, plan_sha256=EXPECTED,
                    protocol=copy.deepcopy(ranker.snapshot), rankings=rows, pool_count=189,
                    top1=sum(r['source_rank'] <= 1 for r in rows), top3=sum(r['source_rank'] <= 3 for r in rows),
                    top5=sum(r['source_rank'] <= 5 for r in rows),
                    fresh_encoder=True, no_external_vector_cache_read=True, query_encoding='target_image_joint')
                result_checks(result, plan, method, group, name)
                freeze_json(path, result)
                print(json.dumps(dict(method=method, group=group['key'], bank=name, stage='retrieved'), ensure_ascii=False), flush=True)
    finally:
        raw.close()
    freeze_json(run / 'methods' / method / 'RETRIEVAL_COMPLETED.json',
        dict(status='completed', plan_sha256=EXPECTED, rankings=360, execution_device='cuda'))


def verify(run, method):
    import numpy as np
    plan = validate(run)
    raw, encoder = fresh_encoder(plan)
    comparisons = ranks = 0
    maximum = 0.0
    bindings = {}
    try:
        for group in plan['methods'][method]['groups']:
            for name in ('clean', 'modified'):
                status(run, 'running', stage='verify', method=method, group=group['key'], bank=name)
                path = output_file(run, method, group['key'], name)
                old = read_json(path)
                result_checks(old, plan, method, group, name)
                if digest(encoder.snapshot) != old['protocol']['model_sha256']:
                    raise ValueError('独立复算编码器身份改变')
                bank, ranker = make_ranker(encoder, group, name)
                for query, previous in zip(group['queries'], old['rankings']):
                    actual = ranker.rank(bank, query['instruction'], query['query_image'])
                    if len(actual) != 189:
                        raise ValueError('复算候选数量不符')
                    for a, b in zip(actual, previous['ranking']):
                        for field in ('Id', 'rank', 'index', 'image_sha256', 'text_sha256',
                                      'query_text_sha256', 'query_image_sha256', 'bank_sha256'):
                            if a[field] != b[field]:
                                raise ValueError('独立复算名次或身份不符：' + field)
                        error = abs(a['score'] - b['score'])
                        if not np.isfinite(error) or error > 1e-6:
                            raise ValueError('独立复算分数超过容差')
                        maximum = max(maximum, error)
                        comparisons += 1
                    ranks += 1
                bindings[str(path)] = sha256_file(path)
                print(json.dumps(dict(method=method, group=group['key'], bank=name, stage='verified'), ensure_ascii=False), flush=True)
    finally:
        raw.close()
    if (comparisons, ranks) != (68040, 360):
        raise ValueError('复算范围未完整覆盖')
    freeze_json(run / 'methods' / method / 'INDEPENDENT_VERIFICATION.json', dict(status='passed',
        fresh_process=True, fresh_encoder=True, execution_device='cuda', no_original_vector_cache_read=True,
        score_comparisons=comparisons, rankings_recomputed=ranks, max_score_error=maximum,
        files=bindings, plan_sha256=EXPECTED))


def resource_state():
    """Read-only resource gate. Existing jobs always retain priority."""
    ps = shutil.which('pwsh') or shutil.which('powershell')
    command = "[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new(); @(Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\\.exe$' } | Select-Object ProcessId,ParentProcessId,CommandLine) | ConvertTo-Json -Compress"
    completed = subprocess.run([ps, '-NoProfile', '-Command', command], capture_output=True,
        text=True, encoding='utf-8', errors='replace', check=True)
    records = json.loads(completed.stdout or '[]')
    if isinstance(records, dict):
        records = [records]
    busy = []
    for record in records:
        cmd = record.get('CommandLine') or ''
        if 'run_groups9_target_image_20261111.py' in cmd or 'litellm' in cmd:
            continue
        busy.append(dict(pid=record['ProcessId'], parent=record['ParentProcessId']))
    locked = []
    for directory in ['groups9_multimodal_20261110', 'lucid_groups9_clean_shopping_20261110']:
        path = PROJECT / 'output' / directory / 'RUN.lock'
        if path.exists():
            try:
                handle = lock(path, create=False)
            except (OSError, RuntimeError):
                locked.append(directory)
            else:
                handle.close()
    return dict(ready=not busy and not locked, python_processes=busy, locked_experiments=locked)


def wait_resources(run):
    while True:
        state = resource_state()
        save_json(run / 'RESOURCE_STATUS.json', dict(checked_at=now(), **state))
        if state['ready']:
            return
        status(run, 'waiting_for_resources', resource_state=state)
        time.sleep(30)


def stage_done(run, method, stage):
    marker = run / 'methods' / method / ('RETRIEVAL_COMPLETED.json' if stage == 'retrieve' else 'INDEPENDENT_VERIFICATION.json')
    if not marker.exists():
        return False
    data = read_json(marker)
    if data['plan_sha256'] != EXPECTED:
        raise ValueError('阶段完成标记绑定错误')
    if stage == 'verify':
        if data['status'] != 'passed' or data['score_comparisons'] != 68040 or data['rankings_recomputed'] != 360:
            raise ValueError('阶段独立复核标记不完整')
        for filename, expected in data['files'].items():
            check(filename, expected)
    return True


def queue(run):
    handle = lock(run / 'RUN.lock')
    try:
        plan = validate(run)
        for method in METHODS:
            for stage in ('retrieve', 'verify'):
                if stage_done(run, method, stage):
                    continue
                wait_resources(run)
                status(run, 'running', method=method, stage=stage)
                logpath = run / 'logs' / f'{method}_{stage}.log'
                logpath.parent.mkdir(parents=True, exist_ok=True)
                command = [sys.executable, '-X', 'utf8', '-B', '-u', str(Path(__file__).resolve()),
                           '--run', str(run), '--stage', stage, '--method', method]
                with logpath.open('a', encoding='utf-8') as log:
                    child = subprocess.Popen(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT)
                    save_json(run / 'ACTIVE_CHILD.json', dict(pid=child.pid, command=command,
                        stage=stage, method=method, launched_at=now()))
                    code = child.wait()
                if code or not stage_done(run, method, stage):
                    raise RuntimeError(f'阶段失败：{method} {stage}，退出码{code}')
        from groups9_target_report import write_report
        write_report(run, plan)
        status(run, 'completed', rankings=720, independent_score_checks=136080)
    finally:
        handle.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=RUN)
    parser.add_argument('--stage', choices=['prepare', 'all', 'retrieve', 'verify', 'report'], default='all')
    parser.add_argument('--method', choices=METHODS)
    args = parser.parse_args()
    run = args.run.resolve()
    if run != RUN.resolve():
        raise ValueError('本轮输出目录必须与冻结计划一致')
    try:
        if args.stage == 'prepare':
            run.mkdir(parents=True, exist_ok=True)
            plan = build_plan(run)
            freeze_json(run / 'SCHEDULED_INPUTS.json', plan)
            freeze_json(run / 'SCOPE_AMENDMENT.json', plan['scope_amendment'])
            print('冻结计划已写入：', run / 'SCHEDULED_INPUTS.json')
            print('PLAN_SHA256=' + sha256_file(run / 'SCHEDULED_INPUTS.json'))
        elif args.stage == 'all':
            queue(run)
        elif args.stage == 'report':
            from groups9_target_report import write_report
            write_report(run, validate(run))
        else:
            if args.method is None:
                raise ValueError('子阶段须指定方法')
            (retrieve if args.stage == 'retrieve' else verify)(run, args.method)
    except Exception as exc:
        detail = dict(status='failed', failed_at=now(), pid=os.getpid(), stage=args.stage,
            method=args.method, error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        (run / 'failures').mkdir(parents=True, exist_ok=True)
        save_json(run / 'failures' / (time.strftime('%Y%m%dT%H%M%S', time.gmtime()) + f'_{os.getpid()}.json'), detail)
        status(run, 'failed', stage=args.stage, method=args.method, error_type=type(exc).__name__)
        raise


if __name__ == '__main__':
    main()
