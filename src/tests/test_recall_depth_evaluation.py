import copy
from pathlib import Path

import pytest

from src.experiments.recall_depth_evaluation import replace_reused_image,validate_submission
from src.vision.image_utils import sha256_file


def entry(product,key,*,error=False):
    return dict(Id=key,FinalASIN=product['id'],Instruction='Buy this public product',
        Actions=['Action: click[Buy Now]'],Completed=True,Success=True,Reward=1.,
        RecordKind='purchase_memory',EntryType='error' if error else 'normal',
        PrimaryImage=dict(ASIN=product['id'],Title=product['title'],LocalPath=product['image'],SHA256=sha256_file(product['image'])))


def test_reused_image_changes_only_designated_entry_and_preserves_full_bank(products,tmp_path):
    bank=[entry(products[0],'source',error=True),entry(products[1],'normal')]
    bank[1]['RetrievalImage']=copy.deepcopy(bank[1]['PrimaryImage'])
    before=copy.deepcopy(bank)
    changed,identifier=replace_reused_image(bank,tmp_path/'bank.json',products[0]['id'],
        products[2]['image'],sha256_file(products[0]['image']))
    assert identifier=='source' and bank==before and len(changed)==len(bank)
    assert changed[1]==before[1]
    assert changed[0]['Actions']==before[0]['Actions'] and changed[0]['PrimaryImage']==before[0]['PrimaryImage']
    assert changed[0]['RetrievalImage']['SHA256']==sha256_file(products[2]['image'])


def test_reused_image_rejects_duplicate_attack_records(products,tmp_path):
    bank=[entry(products[0],'first',error=True),entry(products[0],'second',error=True)]
    with pytest.raises(ValueError,match='唯一指定'):
        replace_reused_image(bank,tmp_path/'bank.json',products[0]['id'],products[2]['image'],sha256_file(products[0]['image']))


def submission(ids):
    return {'PolicyTrace':[dict(SelectedIds=ids,DisplayedMemoryCount=len(ids),AuxiliaryPrompt='fixed guidance',
        Prompt='Additional task guidance:\nfixed guidance\n\nactual context',
        InputImages=[dict(Role='query',SHA256='query-sha')]+[
            dict(Role='recalled_memory',Id=i,SHA256='modified-sha' if i=='source' else 'normal-sha') for i in ids])]}


def test_top_one_uses_global_first_entry_without_forcing_source():
    plan=dict(guidance_text='fixed guidance',query_image_sha256='query-sha',source_memory_id='source',modified_image_sha256='modified-sha')
    ranking=dict(ranking=[{'Id':'normal'},{'Id':'source'},{'Id':'third'}])
    assert validate_submission(submission(['normal']),plan,1,ranking)==['normal']
    with pytest.raises(ValueError,match='全局排名'):
        validate_submission(submission(['source']),plan,1,ranking)


def test_top_three_includes_exact_ranked_images_and_rejects_stale_source_image():
    plan=dict(guidance_text='fixed guidance',query_image_sha256='query-sha',source_memory_id='source',modified_image_sha256='modified-sha')
    ranking=dict(ranking=[{'Id':'normal'},{'Id':'source'},{'Id':'third'}]);record=submission(['normal','source','third'])
    assert validate_submission(record,plan,3,ranking)==['normal','source','third']
    record['PolicyTrace'][0]['InputImages'][2]['SHA256']='stale-original-sha'
    with pytest.raises(ValueError,match='冻结修改图'):
        validate_submission(record,plan,3,ranking)
