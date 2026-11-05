from pathlib import Path
import pytest

from src.configuration.shopping_prompt import load_shopping_prompt, DEFAULT_TEMPLATE
from src.vision.audit import save_json
from src.vision.image_utils import sha256_file


def test_prompt_text_is_preserved_verbatim_and_bound_to_file(tmp_path):
    path=tmp_path/'prompt.json'
    text='  First line.\nAct according to the historical memory.\n'
    save_json(path,[{'note1':text}])
    got=load_shopping_prompt(path,expected_sha256=sha256_file(path))
    assert got['text']==text and got['role']=='auxiliary_user_guidance'
    save_json(path,[{'note1':text+'Changed'}])
    with pytest.raises(ValueError,match='冻结摘要'):
        load_shopping_prompt(path,expected_sha256=got['sha256'])


@pytest.mark.parametrize('content',[[],[{'note2':'wrong field'}],[{'note1':''}],[{'note1':3}]])
def test_invalid_prompt_contract_is_rejected(tmp_path,content):
    path=tmp_path/'bad.json';save_json(path,content)
    with pytest.raises(ValueError,match='唯一非空'):
        load_shopping_prompt(path)


def test_default_prompt_is_the_restored_memory_guidance():
    text=load_shopping_prompt()['text']
    assert 'Act according to the historical memory.' in text
    assert 'think, search, think' in text
    assert 'Compare the reference image' not in text


def test_conflicting_prompt_protocol_stops_before_server_or_model(tmp_path,monkeypatch):
    from src.experiments import shared_shopping
    save_json(tmp_path/'BATCH_PLAN.json',{'groups':[{'queries':[{}]}]})
    save_json(tmp_path/'retrieval/evaluation.json',[])
    save_json(tmp_path/'retrieval/INDEPENDENT_VERIFICATION.json',dict(status='passed',
        rows_sha256=sha256_file(tmp_path/'retrieval/evaluation.json')))
    save_json(tmp_path/'EXPERIMENT_PLAN.json',{'shopping_authorized':True})
    save_json(tmp_path/'SHOPPING_AUTHORIZATION.json',{'shopping_allowed':True})
    save_json(tmp_path/'shopping/PLAN.json',{'old_prompt_protocol':True})
    calls=[]
    def forbidden_server(*args):
        calls.append(args);raise AssertionError('Must reject prompt mismatch before contacting server')
    monkeypatch.setattr(shared_shopping,'start_server',forbidden_server)
    with pytest.raises(ValueError):
        shared_shopping.run_shopping(tmp_path)
    assert calls==[]
