"""冻结成图后的共享长期记忆检索评估、真实复算和报告。"""
import copy
from pathlib import Path

from src.paths import SHARED_MEMORY
from src.memory.shared_pool import load_shared_pool, evaluation_banks
from src.vision.audit import freeze_json, read_json, save_json, digest
from src.vision.image_utils import sha256_file
from src.vision.visual_retrieval import VisualMemoryRetriever, memory_index_text, resolve_image
from src.experiments.defenses import Defender, conditions, table_groups
from src.experiments.query_contract import load_formal_queries


def evaluate(frozen, output, *, pool_path=SHARED_MEMORY, query_manifest,
             source_memory_id=None, device='cuda', encoder_factory=None,
             defense_factory=None, selected_conditions=None, encoder_config=None):
    """只有此模块读取共享库；对照和修改图始终面对同一个完整共享池。"""
    output = Path(output)
    from src.vision.attack_completion import validate_attack_completion
    if frozen.get('status') != 'frozen':
        raise ValueError('只能评估完成且已经冻结的最终图片')
    validate_attack_completion(frozen['metadata'], frozen['requested_steps'])
    # 正式图未就绪时，在读取共享库前退出。
    queries = load_formal_queries(query_manifest, frozen['target'])
    if sha256_file(frozen['image']) != frozen['image_sha256']:
        raise ValueError('冻结成图摘要改变')
    entries = load_shared_pool(pool_path)
    banks, source_id = evaluation_banks(entries, pool_path, frozen['source'],
                                        frozen['image'], source_memory_id=source_memory_id)
    specs = list(selected_conditions if selected_conditions is not None else conditions())
    expected_labels = {s['label']: s for s in conditions()}
    if not specs or len({s['label'] for s in specs}) != len(specs) or any(expected_labels.get(s['label']) != s for s in specs):
        raise ValueError('条件须来自固定防御面板，且不可重复')
    plan = dict(schema_version=1, scope='shared', pool_path=str(Path(pool_path).resolve()),
                pool_sha256=sha256_file(pool_path), pool_count=len(entries),
                source_memory_id=source_id, frozen_image_sha256=frozen['image_sha256'],
                method=frozen['method'], queries=queries, conditions=specs,
                encoder_config=str(Path(encoder_config).resolve()) if encoder_config else None,
                key_image_weight=0.5, query_image_weight=0.5, index_text='title',
                query_image_defended=False, all_memory_images_equally_defended=True,
                shopping_allowed=False,
                record_kinds={kind: sum(e['RecordKind'] == kind for e in entries)
                              for kind in ('purchase_memory', 'simulated_gallery')})
    freeze_json(output / 'evaluation_plan.json', plan)
    if encoder_factory is None:
        encoder_factory = lambda: make_eval_encoder(device, encoder_config)
    encoder = encoder_factory()
    defender = defense_factory(output / 'defended', device) if defense_factory else Defender(output / 'defended', device)
    rows = []
    try:
        for spec in specs:
            for name, original_bank in banks.items():
                bank = copy.deepcopy(original_bank)
                for entry in bank:
                    original = resolve_image(entry, pool_path)
                    image = defender(original, spec)
                    entry['RetrievalImage'] = dict(ASIN=entry['FinalASIN'], LocalPath=str(image.resolve()), SHA256=sha256_file(image))
                bank_file = output / 'banks' / spec['label'] / (name + '.json')
                freeze_json(bank_file, bank)
                ranker = VisualMemoryRetriever(encoder, bank_file, cache_dir=None)
                observations = []
                for query in queries:
                    ranking = ranker.rank(bank, query['instruction'], query['query_image'])
                    if len(ranking) != len(entries):
                        raise ValueError('检索没有覆盖完整共享池')
                    chosen = next(r for r in ranking if r['Id'] == source_id)
                    observations.append(dict(task_id=query['task_id'], rank=chosen['rank'], score=chosen['score']))
                row = dict(condition=spec, bank=name, bank_path=str(bank_file.resolve()),
                           bank_sha256=sha256_file(bank_file), queries=observations,
                           top3=sum(q['rank'] <= 3 for q in observations),
                           top5=sum(q['rank'] <= 5 for q in observations),
                           query_count=len(queries), retrieval_protocol=copy.deepcopy(ranker.snapshot))
                rows.append(row)
                save_json(output / 'evaluation.partial.json', rows)
                print(f"{spec['label']} {name}: 前三 {row['top3']}/{len(queries)}；前五 {row['top5']}/{len(queries)}", flush=True)
        freeze_json(output / 'evaluation.json', rows)
    finally:
        encoder.close()
    # 使用新建编码器重新计算每个条件、每个查询的名次及分数。
    verification = verify(output, encoder_factory=encoder_factory)
    return dict(rows=len(rows), rankings=len(rows) * len(queries), verification=verification)


def make_eval_encoder(device='cuda', config_path=None):
    """默认评估模型独立于两种方法训练时使用的三个代理模型。"""
    from src.paths import CONFIGS, PRIVATE
    from src.vision.dual_rag_bridge import activate, DEFAULT_DUAL_ROOT
    from src.vision.joint_encoder import ChunkedClipEncoder
    private = PRIVATE / 'evaluator.json'
    settings = read_json(config_path or (private if private.is_file() else CONFIGS / 'evaluator.example.json'))
    if settings.get('setting') != 'held_out_public_encoder':
        raise ValueError('当前黑盒评估默认要求独立留出的公开检索器')
    activate()
    from dual_rag.original_attack import resolve_optimization
    from dual_rag.optimization.attack_core.surrogates import resolve_hf_cached_weight
    from dual_rag.encoder import MatchedClipEncoder
    spec = settings['surrogate']
    trained = resolve_optimization(CONFIGS / 'optimization.json')
    if any((m.name, m.pretrained) == (spec['name'], spec['pretrained']) for m in trained.surrogates):
        raise ValueError('正式留出检索器与训练代理相同，不能作为黑盒迁移评估')
    if spec.get('weight_path'):
        weight = Path(spec['weight_path'])
    else:
        weight = next((path for filename in ('open_clip_model.safetensors', 'open_clip_pytorch_model.safetensors')
                       if (path := resolve_hf_cached_weight(spec['hf_repo'], filename=filename)) is not None), None)
    if weight is None or not weight.is_file():
        raise FileNotFoundError('缺少独立留出检索器的本地权重，请先显式准备')
    spec = dict(spec, weight_path=str(weight.resolve()))
    return ChunkedClipEncoder(MatchedClipEncoder(dict(surrogate=spec, surrogate_number=0, device=device), DEFAULT_DUAL_ROOT))


def verify(output, *, encoder_factory=None, device='cuda'):
    output = Path(output)
    plan = read_json(output / 'evaluation_plan.json')
    if sha256_file(plan['pool_path']) != plan['pool_sha256']:
        raise ValueError('基础共享池在评估后改变')
    load_shared_pool(plan['pool_path'])
    rows = read_json(output / 'evaluation.json')
    expected = {(s['label'], bank) for s in plan['conditions'] for bank in ('clean', 'modified')}
    actual = [(r['condition']['label'], r['bank']) for r in rows]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('评估条件缺失或重复')
    encoder = (encoder_factory or (lambda: make_eval_encoder(device)))()
    max_error, count = 0.0, 0
    try:
        for row in rows:
            if row['condition'] not in plan['conditions']:
                raise ValueError('条件的具体参数改变')
            if sha256_file(row['bank_path']) != row['bank_sha256']:
                raise ValueError('评估副本改变')
            bank = read_json(row['bank_path'])
            if len(bank) != plan['pool_count']:
                raise ValueError('评估缩减了共享池')
            ranker = VisualMemoryRetriever(encoder, row['bank_path'], cache_dir=None)
            if digest(encoder.snapshot) != row['retrieval_protocol']['model_sha256']:
                raise ValueError('独立复算编码器的权重或实现身份改变')
            if len(row['queries']) != len(plan['queries']):
                raise ValueError('正式请求数量改变')
            for actual_query, query in zip(row['queries'], plan['queries']):
                if sha256_file(query['query_image']) != query['query_image_sha256']:
                    raise ValueError('正式查询图片改变')
                ranking = ranker.rank(bank, query['instruction'], query['query_image'])
                hit = next(r for r in ranking if r['Id'] == plan['source_memory_id'])
                import math
                if (actual_query['task_id'] != query['task_id'] or type(actual_query['rank']) is not int
                        or actual_query['rank'] != hit['rank'] or not math.isfinite(actual_query['score'])):
                    raise ValueError('独立复算的请求、名次或分数不符')
                error = abs(actual_query['score'] - hit['score'])
                if not math.isfinite(error) or error > 1e-6:
                    raise ValueError('独立复算分数误差超过容差')
                max_error = max(max_error, error)
                count += 1
            for k in (3, 5):
                if row[f'top{k}'] != sum(q['rank'] <= k for q in row['queries']):
                    raise ValueError('汇总与逐请求名次不符')
    finally:
        encoder.close()
    result = dict(status='passed', rows=len(rows), rankings_recomputed=count,
                  max_score_error=max_error, evaluated_file_sha256=sha256_file(output / 'evaluation.json'),
                  evaluation_plan_sha256=sha256_file(output / 'evaluation_plan.json'),
                  fresh_encoder=True, recomputed_embeddings_and_rankings=True)
    freeze_json(output / 'independent_verification.json', result)
    return result


def report(output):
    output = Path(output)
    plan = read_json(output / 'evaluation_plan.json')
    rows = read_json(output / 'evaluation.json')
    verified = read_json(output / 'independent_verification.json')
    if (verified.get('status') != 'passed'
            or verified['evaluated_file_sha256'] != sha256_file(output / 'evaluation.json')
            or verified['evaluation_plan_sha256'] != sha256_file(output / 'evaluation_plan.json')):
        raise ValueError('缺少绑定当前结果的独立复算证据')
    lookup = {(r['condition']['label'], r['bank']): r for r in rows}
    lines = ['# 共享长期记忆检索评估', '',
             f"方法：{plan['method']}。共享池 {plan['pool_count']} 条；真实购买记忆 {plan['record_kinds']['purchase_memory']} 条，模拟图库 {plan['record_kinds']['simulated_gallery']} 条。",
             '', '正式查询使用模糊文字及单独生成、经核对无明确名称和品牌的图片。所有图库图片接受相同防御，查询图片保持原样。',
             '', '前三及前五表示检索命中，不表示购买成功。本轮没有购物或外部模型调用。', '',
             '| 条件 | 原图前三 | 修改图前三 | 原图前五 | 修改图前五 |', '|---|---:|---:|---:|---:|']
    for title, labels in table_groups():
        if not all((lbl, b) in lookup for lbl in labels for b in ('clean', 'modified')):
            continue
        values = []
        for k in (3, 5):
            for name in ('clean', 'modified'):
                qs = [q for lbl in labels for q in lookup[(lbl, name)]['queries']]
                values.append(f"{sum(q['rank'] <= k for q in qs)}/{len(qs)}")
        lines.append('| ' + title + ' | ' + ' | '.join(values) + ' |')
    lines += ['', '## 全部条件与种子', '', '| 条件 | 图库 | 前三 | 前五 |', '|---|---|---:|---:|']
    for row in rows:
        lines.append(f"| {row['condition']['label']} | {row['bank']} | {row['top3']}/{row['query_count']} | {row['top5']}/{row['query_count']} |")
    lines += ['', f"独立重新编码并复算 {verified['rankings_recomputed']} 次排名，最大分数误差 {verified['max_score_error']}。", '']
    path = output / 'REPORT.md'
    path.write_text('\n'.join(lines), encoding='utf-8')
    return path
