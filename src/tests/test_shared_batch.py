import copy
from pathlib import Path

from PIL import Image
import pytest

from src.vision.audit import read_json,save_json
from src.vision.image_utils import sha256_file
from src.experiments.shared_batch import replace_source_records,make_banks,evaluate,verify,OLD_ASINS
from src.experiments.shared_shopping import policy_references
from test_shared_pool import ToyEncoder


@pytest.fixture
def batch_inputs(products,tmp_path):
    entries=[]
    for index,product in enumerate(products):
        entries.append(dict(Id='memory_'+str(index),FinalASIN=product['id'],Completed=True,
            RecordKind='purchase_memory',Instruction='Observed purchase '+str(index),Actions=['Observed action'],
            Reward=1.,Success=True,PrimaryImage=dict(ASIN=product['id'],Title=product['title'],
                LocalPath=product['image'],SHA256=product['image_sha256'])))
    groups=[]
    for key,index,color in [('02_chair',1,'purple'),('09_toothbrush',2,'orange')]:
        image=tmp_path/(key+'.png');Image.new('RGB',(24,24),color).save(image)
        entries[index].update(SourceGroup=key,EntryType='error')
        groups.append(dict(key=key,source=products[index],target=products[0],
            frozen_image=str(image),frozen_image_sha256=sha256_file(image),
            queries=[dict(task_id=f'query_{n}',instruction=f'Generic instruction {n}',
                query_image=products[0]['image'],query_image_sha256=products[0]['image_sha256']) for n in range(2)]))
    pool=tmp_path/'shared_memory.json'
    save_json(pool,dict(schema_version=1,scope='shared',entries=entries))
    return entries,groups,pool


def test_two_old_source_records_are_replaced_without_changing_other_history(batch_inputs):
    entries,_,_=batch_inputs
    old=copy.deepcopy(entries)
    for row in old[1:]:row['FinalASIN']=OLD_ASINS[row['SourceGroup']]
    updated,removed=replace_source_records(old,entries[1:])
    assert updated==entries and len(removed)==2
    assert updated[0]==old[0]
    assert old[1]['FinalASIN']==OLD_ASINS['02_chair']


def test_replacement_refuses_an_unexpected_old_product(batch_inputs):
    entries,_,_=batch_inputs
    with pytest.raises(ValueError,match='唯一条目'):
        replace_source_records(entries,entries[1:])


def test_both_images_are_substituted_in_one_bank_and_histories_stay_intact(batch_inputs):
    entries,groups,pool=batch_inputs
    original=copy.deepcopy(entries)
    banks,ids=make_banks(entries,groups,pool)
    assert entries==original and len(ids)==2
    assert banks['clean'][0]==banks['modified'][0]
    for group in groups:
        row=next(e for e in banks['modified'] if e['Id']==ids[group['key']])
        assert row['RetrievalImage']['SHA256']==group['frozen_image_sha256']
        assert row['FinalASIN']==group['source']['id']
        source=next(e for e in banks['clean'] if e['Id']==row['Id'])
        assert {k:v for k,v in row.items() if k not in ('RetrievalImage','ImageSubstitution')}==source


def test_new_image_cannot_remain_bound_to_an_old_title(batch_inputs):
    entries,groups,pool=batch_inputs
    entries[1]['PrimaryImage']['Title']='Old unrelated title'
    with pytest.raises(ValueError,match='名称或原图'):
        make_banks(entries,groups,pool)


def test_gallery_policy_reference_does_not_invent_a_purchase(batch_inputs):
    entries,_,_=batch_inputs
    entries[0].update(RecordKind='simulated_gallery',Completed=False)
    entries[0].pop('Actions');entries[0].pop('Instruction')
    adapted=policy_references(entries)
    assert adapted[0]['Completed'] is False and adapted[0]['Actions']==[]
    assert 'no observed purchase' in adapted[0]['Instruction']
    assert adapted[1:]==entries[1:] and 'Actions' not in entries[0]


def test_shared_retrieval_recomputes_both_vectors_and_independently_verifies(batch_inputs,tmp_path):
    entries,groups,pool=batch_inputs
    banks,ids=make_banks(entries,groups,pool)
    poisoned=tmp_path/'poisoned.json';save_json(poisoned,dict(entries=banks['modified']))
    specs=[dict(label='none',kind='none',size=512)]
    plan=dict(groups=groups,pool_path=str(pool),pool_sha256=sha256_file(pool),pool_count=len(entries),
        poisoned_pool=str(poisoned),poisoned_pool_sha256=sha256_file(poisoned),
        source_memory_ids=ids,conditions=specs,record_kinds={'purchase_memory':3,'simulated_gallery':0})
    save_json(tmp_path/'BATCH_PLAN.json',plan)
    created=[]
    def encoder():
        value=ToyEncoder();created.append(value);return value
    rows=evaluate(tmp_path,encoder_factory=encoder,defense_factory=lambda path:lambda image,spec:Path(image),specs=specs)
    assert len(rows)==4 and len(created)==2 and all(e.closed for e in created)
    proof=read_json(tmp_path/'retrieval/INDEPENDENT_VERIFICATION.json')
    assert proof['status']=='passed' and proof['rankings']==8 and proof['max_score_error']==0
    clean=read_json(tmp_path/'retrieval/indexes/none/clean.json')
    modified=read_json(tmp_path/'retrieval/indexes/none/modified.json')
    assert len(clean['vectors'])==len(modified['vectors'])==len(entries)
    assert clean['identity']['bank_sha256']!=modified['identity']['bank_sha256']
    changed=read_json(tmp_path/'retrieval/evaluation.json');changed[0]['queries'][0]['rank']+=1
    save_json(tmp_path/'retrieval/evaluation.json',changed)
    with pytest.raises(ValueError,match='名次不一致'):
        verify(tmp_path,encoder_factory=encoder,specs=specs)
