"""Read existing evidence only; no encoder, generation, or shopping calls."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
INPUTS = {}


def read(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    raw = path.read_bytes()
    INPUTS[str(path.resolve())] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw.decode('utf-8-sig'))


def entries(path):
    obj = read(path)
    return obj if isinstance(obj, list) else obj['entries']


def stats(ranks):
    return dict(n=len(ranks), mean=mean(ranks), minimum=min(ranks),
                maximum=max(ranks), top1=sum(x == 1 for x in ranks),
                top3=sum(x <= 3 for x in ranks), top5=sum(x <= 5 for x in ranks))


plan = read('output/groups9_multimodal_20261110/EXPERIMENT_PLAN.json')
old100 = entries('../lage/src/output/lage_runs/catalog_shared_memory_20261103/shared_memory_pool.json')
old189 = entries('output/shared_memory/pool_189_v1/shared_memory.json')
new189 = entries(plan['pool'])
sets = {'old100': {e['FinalASIN'] for e in old100},
        'old189': {e['FinalASIN'] for e in old189}}


def signature(entry):
    return entry['FinalASIN'], entry['PrimaryImage']['SHA256']


old_normals = [e for e in old189 if e.get('EntryType') == 'normal']
new_normals = [e for e in new189 if e.get('EntryType') == 'normal']
pool_comparison = dict(
    old100_count=len(old100), old189_count=len(old189), new189_count=len(new189),
    old189_current_asin_overlap=len(sets['old189'] & {e['FinalASIN'] for e in new189}),
    normal_counts=[len(old_normals), len(new_normals)],
    normal_asin_and_declared_image_hash_multisets_identical=(
        Counter(map(signature, old_normals)) == Counter(map(signature, new_normals))))

current = []
source_evidence = None
for method, obj in plan['methods'].items():
    group = next(g for g in obj['groups'] if g['key'] == '07_shampoo')
    bank = entries(group['banks']['clean']['path'])
    byid = {e['Id']: e for e in bank}
    source = byid[group['source_memory_id']]
    source_evidence = dict(source=group['source'], target=group['target'],
        current_source_attributes=source.get('Attributes'),
        current_source_feature_text=source.get('FeatureText'),
        query_examples=group['queries'][:3])
    for kind in ['clean', 'modified']:
        result = read(f'output/groups9_multimodal_20261110/methods/{method}/retrieval/07_shampoo/{kind}.json')
        row = dict(method=method, bank=kind,
                   joint=stats([q['source_rank'] for q in result['rankings']]))
        restricted = {}
        for label, asin_set in sets.items():
            ranks = []
            counts = set()
            for query in result['rankings']:
                kept = [e for e in query['ranking']
                        if byid[e['Id']]['FinalASIN'] in asin_set
                        or e['Id'] == group['source_memory_id']]
                counts.add(len(kept))
                ranks.append(next(i + 1 for i, e in enumerate(kept)
                                  if e['Id'] == group['source_memory_id']))
            restricted[label] = dict(retained_counts=sorted(counts), **stats(ranks))
        row['restricted_current_scores'] = restricted
        historical_method = 'lucid' if method == 'lucid' else 'PerLAGE_no_pixel'
        historical_root = 'lucid_groups9_20261109' if method == 'lucid' else 'perlage_no_pixel_groups9_20261109'
        old_results = read(f'output/{historical_root}/groups/07_shampoo/methods/{historical_method}/evaluation/evaluation.json')
        old_result = next(x for x in old_results if x['bank'] == kind)
        row['previous_text'] = stats([q['rank'] for q in old_result['queries']])
        row['current_image_sha256'] = (group['source']['image_sha256'] if kind == 'clean'
                                       else group['frozen_image_sha256'])
        row['first_query_top5'] = [dict(rank=i + 1, asin=byid[e['Id']]['FinalASIN'],
            title=byid[e['Id']]['PrimaryImage']['Title'], score=e['score'],
            in_old100=byid[e['Id']]['FinalASIN'] in sets['old100'],
            in_old189=byid[e['Id']]['FinalASIN'] in sets['old189'])
            for i, e in enumerate(result['rankings'][0]['ranking'][:5])]
        current.append(row)

nov6 = read('output/shampoo_k13_20261106/retrieval/RANKINGS.json')
nov7 = read('output/shampoo_ablation_20261107/evaluation/RANKINGS.json')
nov7plan = read('output/shampoo_ablation_20261107/evaluation/PLAN.json')
nov6plan = read('output/shampoo_k13_20261106/PLAN.json')
history = dict(
    november6=stats([q['source_rank'] for q in nov6['rows']]),
    november7={r['variant']: stats([q['source_rank'] for q in r['queries']]) for r in nov7['rows']},
    same_legacy_image=(nov6plan['modified_image_sha256'] == nov7plan['variants']['perlage_legacy_reference']['sha256']),
    same_reference_image=(nov6plan['query_image_sha256'] == nov7plan['query_image_sha256']),
    current_and_legacy_images_identical={r['method']: r['current_image_sha256'] == nov6plan['modified_image_sha256']
                                         for r in current if r['bank'] == 'modified'})

# Validate that all source evidence stayed byte-identical during this analysis.
for filename, digest in INPUTS.items():
    assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == digest
audit = dict(created_at=datetime.now(timezone.utc).isoformat(),
    scope='posthoc_saved_score_analysis_only', encoder_calls=0,
    generation_calls=0, shopping_calls=0, formal_files_modified=False,
    source_evidence=source_evidence, pool_comparison=pool_comparison,
    historical_checks=history, current_results=current,
    limitations=[
        'Restricted rankings reuse current scores and only remove absent candidates; they do not reproduce old model, text keys, query, or images.',
        'The old100 intersection retains 87 current records, not the complete historical 100-record pool.',
        'Metadata image hashes identify background overlap; no new encoder factor intervention was run.',
        'Historical and current images differ; remaining factor contributions cannot be separately quantified from these comparisons.'],
    input_sha256=INPUTS, inputs_unchanged_after_analysis=True)
(OUT / 'AUDIT.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(pool_comparison=pool_comparison, historical_checks=history,
    current=[{k:v for k,v in r.items() if k not in ['first_query_top5']} for r in current]), ensure_ascii=False))
