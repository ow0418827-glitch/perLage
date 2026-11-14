"""Read-only arithmetic audit of frozen experiment outputs; no model calls."""
from __future__ import annotations
import collections, copy, hashlib, json, math, pathlib, statistics

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = pathlib.Path(__file__).resolve().parent
RUNS = {'perlage': ('perlage_no_pixel_groups9_20261109', 'PerLAGE_no_pixel'),
        'lucid': ('lucid_groups9_20261109', 'lucid')}

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def metrics(rows):
    ranks = [r['rank'] for r in rows]
    return {'n':len(ranks), **{f'top{k}':sum(r <= k for r in ranks) for k in (1,3,5)},
            'mrr':statistics.mean(1/r for r in ranks), 'mean_rank':statistics.mean(ranks),
            'min_rank':min(ranks), 'max_rank':max(ranks), 'mean_score':statistics.mean(r['score'] for r in rows)}

def count_shopping(rows):
    return {'n':len(rows), 'completed':sum(x['completed'] for x in rows),
            'source':sum(x['source_purchase'] for x in rows),
            'target':sum(x['target_purchase'] for x in rows),
            'other':sum(x['completed'] and not x['source_purchase'] and not x['target_purchase'] for x in rows),
            'failed':sum(not x['completed'] for x in rows),
            'mean_reward':statistics.mean(x['reward'] for x in rows),
            'source_recalled':sum(x['source_recalled'] for x in rows),
            'source_recalled_and_bought':sum(x['source_recalled'] and x['source_purchase'] for x in rows),
            'source_not_recalled_bought':sum(not x['source_recalled'] and x['source_purchase'] for x in rows)}

def paired(a,b,ka,kb):
    counts=collections.Counter((bool(x[ka]),bool(y[kb])) for x,y in zip(a,b))
    left_only=counts[(True,False)]; right_only=counts[(False,True)]; n=left_only+right_only
    # Descriptive only: 20 wording variants per group are clustered, not 180 independent products.
    p=min(1,2*sum(math.comb(n,k) for k in range(min(left_only,right_only)+1))/2**n) if n else 1
    return {'both':counts[(True,True)],'left_only':left_only,'right_only':right_only,
            'neither':counts[(False,False)],'naive_exact_mcnemar_p_not_cluster_adjusted':p}

audit={'audit_scope':'Arithmetic and payload audit of saved records only; no model inference or rerun.',
       'caveat':'Nine product pairs, twenty wording variants each. No independent-product significance claimed.',
       'methods':{},'comparison':{},'checks':[]}
raw={}
for tag,(directory,method) in RUNS.items():
    base=ROOT/'output'/directory; groups={}; all_rows={'clean':[],'modified':[]}; shopping=[]
    outcomes=read(base/'shopping/outcomes.json'); outcome_map={(x['group'],x['task_id']):x for x in outcomes}
    aggregate_issues=[]; actual_devices=collections.Counter(); request_configs=collections.Counter(); auxiliary=collections.Counter()
    input_roles=collections.Counter(); source_eval_disagreements=[]; state_disagreements=[]; target_presence={}
    episode_trace_count=0; requests=[]
    for gp in sorted((base/'groups').iterdir()):
        if not gp.is_dir(): continue
        ev=gp/'methods'/method/'evaluation'; plan=read(ev/'evaluation_plan.json'); rows=read(ev/'evaluation.json')
        gr={}; rowlookup={}
        for row in rows:
            bank=row['bank']; all_rows[bank]+=row['queries']; gr[bank]=metrics(row['queries'])
            rowlookup[bank]={x['task_id']:x for x in row['queries']}
            for k in (1,3,5):
                if gr[bank][f'top{k}']!=row[f'top{k}']: aggregate_issues.append([gp.name,bank,f'top{k}'])
        v=read(ev/'independent_verification.json'); gr['independent_verification']=v
        bank=read(base/f'shopping/policy_banks/{gp.name}_modified.json'); byid={x['Id']:x for x in bank}
        source=byid[plan['source_memory_id']]['FinalASIN']
        definition=read(ROOT/f'output/experiment_base_groups9_20261109/groups/{gp.name}/definition.json')
        target=definition['target']['id']
        target_presence[gp.name]=[x['Id'] for x in bank if x.get('FinalASIN')==target]
        gr['source_asin']=source; gr['target_asin']=target
        episodes=[]
        for q in plan['queries']:
            task=q['task_id']; key=(gp.name,task); ep=read(base/f'shopping/episodes/{gp.name}/{task}.json'); ot=outcome_map[key]
            tr=ep['RetrievalTrace'][0]; ranking=tr['Ranking']; rr=next(x for x in ranking if x['Id']==plan['source_memory_id'])
            selected=ep['PolicyTrace'][0]['SelectedIds']; completed=ep['Completed']; final=ep.get('FinalASIN')
            row={'group':gp.name,'task_id':task,'instruction':ep['Instruction'],'completed':completed,
                 'final_asin':final,'source_purchase':completed and final==source,'target_purchase':completed and final==target,
                 'reward':ep['EnvironmentReward'],'source_recalled':plan['source_memory_id'] in selected,
                 'source_rank':rr['rank'],'source_score':rr['score'],'selected_ids':selected,
                 'clean_source_rank':rowlookup['clean'][task]['rank'],
                 'source_score_cpu_gpu_difference':rr['score']-rowlookup['modified'][task]['score']}
            if rr['rank']!=rowlookup['modified'][task]['rank']: source_eval_disagreements.append(key)
            for field,value in [('completed',completed),('final_asin',final),('misleading_purchase',row['source_purchase']),('exact_target_purchase',row['target_purchase'])]:
                if ot[field]!=value:state_disagreements.append([*key,field])
            for trace in ep['PolicyTrace']:
                request_configs[json.dumps(trace['PolicyRequest'],sort_keys=True)]+=1
                auxiliary[trace['AuxiliaryPrompt']]+=1
                input_roles.update(x.get('Role','none') for x in trace.get('InputImages',[]))
                requests.append(trace['PolicyRequest'])
            actual_devices[tr['RetrievalProtocol']['encoder']['surrogate']['device']]+=1
            episode_trace_count+=len(ep['PolicyTrace']); episodes.append(row); shopping.append(row)
        gr['shopping']=count_shopping(episodes); gr['episodes']=episodes; gr['plan']=plan
        groups[gp.name]=gr
    audit['methods'][tag]={'path':str(base),'status':read(base/'RUN_STATUS.json'),'n_outcomes':len(outcomes),'unique_keys':len(outcome_map),
       'retrieval':{b:metrics(rows) for b,rows in all_rows.items()}, 'shopping':count_shopping(shopping),'groups':groups,
       'aggregate_issues':aggregate_issues,'state_disagreements':state_disagreements,
       'shopping_rank_disagreements_with_offline':source_eval_disagreements,
       'max_absolute_shopping_source_score_difference_from_offline':max(abs(x['source_score_cpu_gpu_difference']) for x in shopping),
       'shopping_encoder_devices':dict(actual_devices),'policy_trace_count':episode_trace_count,
       'policy_request_configs':[{'config':json.loads(k),'n':v} for k,v in request_configs.items()],
       'auxiliary_prompts':dict(auxiliary),'input_image_roles':dict(input_roles), 'target_product_entries_in_pool':target_presence}
    raw[tag]={'shopping':shopping,'retrieval':all_rows,'groups':groups}

p=raw['perlage']; l=raw['lucid']; ps=p['shopping']; ls=l['shopping']
assert [(x['group'],x['task_id'],x['instruction']) for x in ps]==[(x['group'],x['task_id'],x['instruction']) for x in ls]
audit['comparison']['identical_task_key_and_text_pairs']=len(ps)
audit['comparison']['source_purchase_paired_perlage_left_lucid_right']=paired(ps,ls,'source_purchase','source_purchase')
audit['comparison']['target_purchase_paired_perlage_left_lucid_right']=paired(ps,ls,'target_purchase','target_purchase')
audit['comparison']['same_final_product']=sum(x['final_asin']==y['final_asin'] for x,y in zip(ps,ls))
audit['comparison']['same_ordered_three_memories']=sum(x['selected_ids']==y['selected_ids'] for x,y in zip(ps,ls))
audit['comparison']['same_three_memories_as_set']=sum(set(x['selected_ids'])==set(y['selected_ids']) for x,y in zip(ps,ls))
audit['comparison']['same_ordered_three_memories_different_final_product']=sum(x['selected_ids']==y['selected_ids'] and x['final_asin']!=y['final_asin'] for x,y in zip(ps,ls))
audit['comparison']['same_three_memories_set_different_final_product']=sum(set(x['selected_ids'])==set(y['selected_ids']) and x['final_asin']!=y['final_asin'] for x,y in zip(ps,ls))
audit['comparison']['perlage_source_bought_with_clean_top3']=sum(x['source_purchase'] and x['clean_source_rank']<=3 for x in ps)
audit['comparison']['perlage_source_bought_without_clean_top3']=sum(x['source_purchase'] and x['clean_source_rank']>3 for x in ps)
audit['comparison']['perlage_new_top3_rows']=[{'group':x['group'],'task_id':x['task_id'],'clean_rank':x['clean_source_rank'],'modified_rank':x['source_rank'],'source_purchase':x['source_purchase']} for x in ps if x['source_rank']<=3 and x['clean_source_rank']>3]
audit['comparison']['recall_strata']={}
for name,fn in [('both_recalled',lambda x,y:x['source_recalled'] and y['source_recalled']),
                ('only_lucid_recalled',lambda x,y:not x['source_recalled'] and y['source_recalled']),
                ('neither_recalled',lambda x,y:not x['source_recalled'] and not y['source_recalled'])]:
    pairs=[(x,y) for x,y in zip(ps,ls) if fn(x,y)]
    audit['comparison']['recall_strata'][name]={'n':len(pairs),
        'perlage_source_purchase':sum(x['source_purchase'] for x,y in pairs),
        'lucid_source_purchase':sum(y['source_purchase'] for x,y in pairs),
        'lucid_source_purchase_by_group':dict(collections.Counter(y['group'] for x,y in pairs if y['source_purchase']))}
audit['comparison']['rank_win_tie_loss_perlage_vs_lucid']={
    'perlage_better':sum(x['source_rank']<y['source_rank'] for x,y in zip(ps,ls)),
    'tie':sum(x['source_rank']==y['source_rank'] for x,y in zip(ps,ls)),
    'lucid_better':sum(x['source_rank']>y['source_rank'] for x,y in zip(ps,ls))}
bank_checks=[]; plan_checks=[]
for group in p['groups']:
    pg=p['groups'][group]; lg=l['groups'][group]
    pp=copy.deepcopy(pg['plan']); lp=copy.deepcopy(lg['plan'])
    plan_checks.append({'group':group,'different_fields':[k for k in pp.keys()|lp.keys() if pp.get(k)!=lp.get(k)]})
    pb=read(ROOT/f'output/{RUNS["perlage"][0]}/shopping/policy_banks/{group}_modified.json')
    lb=read(ROOT/f'output/{RUNS["lucid"][0]}/shopping/policy_banks/{group}_modified.json')
    diffs=[{'id':a['Id'],'fields':[k for k in a.keys()|b.keys() if a.get(k)!=b.get(k)]} for a,b in zip(pb,lb) if a!=b]
    bank_checks.append({'group':group,'count_perlage':len(pb),'count_lucid':len(lb),'differences':diffs})
    gp=pg['episodes']; gl=lg['episodes']
    pg['paired_source_purchases_vs_lucid']=paired(gp,gl,'source_purchase','source_purchase')
audit['comparison']['plan_differences']=plan_checks
audit['comparison']['policy_bank_differences']=bank_checks
audit['comparison']['clean_rank_score_rows_identical']=p['retrieval']['clean']==l['retrieval']['clean']
audit['comparison']['source_top3_gain_perlage']=audit['methods']['perlage']['retrieval']['modified']['top3']-audit['methods']['perlage']['retrieval']['clean']['top3']
audit['comparison']['source_top3_gain_lucid']=audit['methods']['lucid']['retrieval']['modified']['top3']-audit['methods']['lucid']['retrieval']['clean']['top3']

clean=ROOT/'output/lucid_groups9_clean_shopping_20261110'
co=read(clean/'shopping/outcomes.json')
audit['clean_control_incomplete_snapshot']={'status':read(clean/'RUN_STATUS.json'),'episodes':len(co),
    'source':sum(x.get('source_purchase',x.get('misleading_purchase',False)) for x in co),
    'target':sum(x['exact_target_purchase'] for x in co),'complete':len(co)==180,
    'note':'Live queue; snapshot can advance after this audit. Do not use partial control as a full-experiment baseline.'}
(OUT/'metrics_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
small={'methods':{k:{x:v[x] for x in ('retrieval','shopping','shopping_encoder_devices','shopping_rank_disagreements_with_offline','max_absolute_shopping_source_score_difference_from_offline')} for k,v in audit['methods'].items()},'comparison':{k:v for k,v in audit['comparison'].items() if k not in ('policy_bank_differences','plan_differences')},'clean_control':audit['clean_control_incomplete_snapshot']}
print(json.dumps(small,ensure_ascii=False,indent=2))
