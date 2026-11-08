import copy
import json

import numpy as np
from PIL import Image
import pytest

from src.paths import ProjectPath as Path, PROJECT, OUTPUT, PROMPTS, relocate_project_path
from src.visual.audit import save_json
from src.visual.image_utils import sha256_file
from src.agent.prompts.loader import load_shopping_prompt
from src.agent.memory.balanced_pool import GROUP_PATTERNS, relevant_public_title, validate_balanced_pool


def test_historical_locations_resolve_only_inside_current_project():
    previous=PROJECT/'src/output/example/ready.json'
    assert Path(previous)==OUTPUT/'example/ready.json'
    other=PROJECT.parent/'lage/src/output/example/ready.json'
    assert relocate_project_path(other)==str(other)
    assert Path(PROJECT/'src/data/indication_prompt_template_neutral.json')==PROMPTS/'neutral.json'


def test_prompt_profile_changes_actual_text_without_editing_execution_code(tmp_path):
    first=tmp_path/'first.json';second=tmp_path/'second.json'
    save_json(first,[dict(note1='First guidance.\n')]);save_json(second,[dict(note1='Second guidance.')])
    settings=tmp_path/'settings.json'
    save_json(settings,dict(default_profile='first',profiles=dict(first='first.json',second='second.json')))
    original=load_shopping_prompt(settings_path=settings)
    assert original['text']=='First guidance.\n'
    assert load_shopping_prompt(profile='second',settings_path=settings)['text']=='Second guidance.'
    save_json(settings,dict(default_profile='second',profiles=dict(first='first.json',second='second.json')))
    changed=load_shopping_prompt(settings_path=settings)
    assert changed['text']=='Second guidance.' and changed['settings_sha256']!=original['settings_sha256']
    assert changed['sha256']==sha256_file(second)


def test_latent_only_mode_removes_every_pixel_stage_and_fixed_residual(products):
    from src.experiments.pipeline import generation_definition
    from src.visual._perlage.pipeline import validate_definition,initialization_config
    definition=generation_definition(dict(source=products[1],target=products[0]),{})
    config=validate_definition(definition);initial=initialization_config(definition)
    assert config.pixel_optimization is False and config.latent['auxiliary_levels']==0
    assert initial.pixel_clean_steps==initial.pixel_diffusion_steps==initial.pixel_levels==0
    assert initial.latent_clean_steps==initial.latent_diffusion_steps==40
    for values in [dict(latent=dict(auxiliary_levels=4)),dict(initialization=dict(optimizer=dict(pixel_clean_steps=20)))]:
        with pytest.raises(ValueError,match='像素'):
            validate_definition(dict(definition,**values))
    legacy=dict(definition,pixel_optimization=True,latent=dict(auxiliary_levels=4))
    assert initialization_config(legacy).pixel_diffusion_steps==60
    assert validate_definition(legacy).latent['auxiliary_levels']==4
    assert validate_definition(dict(definition,pixel_optimization=True)).latent['auxiliary_levels']==4


@pytest.mark.parametrize('group,title',[
    ('03_cookies','Cookie Box Packaging for Homemade Gifts'),
    ('07_shampoo','Empty Refillable Shampoo Dispenser Bottle'),
    ('09_toothbrush','Travel Charger Base for Electric Toothbrush'),
    ('08_speaker','Power Cord for Wireless Speaker'),
])
def test_public_memory_precheck_excludes_name_mentions_in_accessories(group,title):
    assert not relevant_public_title(group,title)


def test_global_pool_requires_all_nine_quotas_unique_images_and_honest_origins(tmp_path):
    entries=[]
    for group in GROUP_PATTERNS:
        for position in range(21):
            i=len(entries);image=tmp_path/f'{i}.png'
            Image.new('RGB',(16,16),(i,i*2%256,i*3%256)).save(image)
            asin=f'A{i:09d}'
            entries.append(dict(Id=str(i),FinalASIN=asin,SourceGroup=group,
                EntryType='normal' if position<20 else 'error',RecordKind='simulated_gallery',Completed=False,
                PrimaryImage=dict(ASIN=asin,Title=f'Public product {i}',LocalPath=str(image),SHA256=sha256_file(image))))
    path=tmp_path/'shared_memory.json';pool=dict(schema_version=1,scope='shared',entries=entries)
    save_json(path,pool)
    assert len(validate_balanced_pool(path))==189
    missing=copy.deepcopy(pool);missing['entries'].pop();save_json(path,missing)
    with pytest.raises(ValueError,match='每组二十'):
        validate_balanced_pool(path)
    dishonest=copy.deepcopy(pool);dishonest['entries'][0]['Completed']=True;save_json(path,dishonest)
    with pytest.raises(ValueError,match='模拟图库'):
        validate_balanced_pool(path)
    duplicate=copy.deepcopy(pool);duplicate['entries'][1]['PrimaryImage'].update(
        LocalPath=entries[0]['PrimaryImage']['LocalPath'],SHA256=entries[0]['PrimaryImage']['SHA256'])
    save_json(path,duplicate)
    with pytest.raises(ValueError,match='重复图片'):
        validate_balanced_pool(path)


def test_pixel_pipeline_improves_target_similarity_with_strict_saved_budget(tmp_path):
    import torch
    import torch.nn.functional as F
    from src.visual.encoders.backbone import ImageFeatures
    from src.visual._lucid.config import AttackConfig,SurrogateConfig,ProjectConfig,AcceptanceConfig
    from src.visual._lucid.pipeline import run_attack

    class ToyModel:
        device='cpu'
        def __init__(self,index):self.index=index;self.cache_identity=str(index)
        def snapshot(self):return dict(toy_model=self.index)
        def encode_image(self,image):
            channels=image.mean((-2,-1))*image.new_tensor([1,.7+self.index/10,1.2])
            global_embedding=F.normalize(channels,dim=-1)
            patches=global_embedding[:,None,:].expand(-1,4,-1)
            return ImageFeatures(global_embedding,patches)
    source=tmp_path/'source.png';target=tmp_path/'target.png'
    Image.new('RGB',(24,24),(50,100,150)).save(source);Image.new('RGB',(24,24),(200,80,30)).save(target)
    config=ProjectConfig(attack=AttackConfig(epsilon=1/255,reference_steps=6,clusters=2,
        cluster_iterations=2,sinkhorn_iterations=2,text_weight=0),
        surrogates=tuple(SurrogateConfig('toy','fixture') for _ in range(3)),device='cpu',
        acceptance=AcceptanceConfig(decoded_max_pixel_difference=1))
    result=run_attack(source,target,None,config,steps=6,output_directory=tmp_path/'run',
                      surrogates=[ToyModel(i) for i in range(3)])
    pixels=np.asarray(Image.open(tmp_path/'run/adversarial.png')).astype(int)
    assert np.abs(pixels-np.array([50,100,150])).max()==1
    assert result.metadata['attack']['steps_completed']==6
    assert result.metadata['decoded_image_audit']['within_budget'] is True
    model=ToyModel(0);a=result.decoded_tensor;b=a.new_tensor([200,80,30])[None,:,None,None]/255
    before=a.new_tensor([50,100,150])[None,:,None,None]/255
    similarity=lambda x:float((model.encode_image(x).global_embedding*model.encode_image(b).global_embedding).sum())
    assert similarity(a)>similarity(before)


def test_webshop_client_uses_instance_service_url_without_global_instruction_file(monkeypatch):
    from src.webshop import client
    calls=[]
    class Response:
        text='<html><body><h1>Instruction: buy a product</h1><button>Search</button></body></html>'
        url='http://127.0.0.1:8123/example'
        def raise_for_status(self):pass
    def get(url,**kwargs):calls.append(url);return Response()
    monkeypatch.setattr(client.requests,'get',get)
    env=client.webshopEnv(base_url='http://127.0.0.1:8123/')
    observation,_,_,_=env.step('example','reset')
    assert calls==['http://127.0.0.1:8123/example']
    assert 'buy a product' in observation


def test_configured_guidance_reaches_actual_policy_payload_verbatim(tmp_path):
    from types import SimpleNamespace
    from src.agent import shopping
    requests=[]
    def create(**kwargs):
        requests.append(kwargs)
        return {'choices':[{'message':{'content':'click[Buy Now]'},'finish_reason':'stop'}]}
    client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    class FixtureEnvironment:
        sessions={}
        def step(self,session,action):
            if action=='reset':
                return 'Instruction: fixture request',0,False,dict(instruction='fixture request')
            return 'Fixture finished',1,True,dict(instruction='fixture request',final_asin='fixture')
    template=tmp_path/'guidance.json'
    texts=['Configuration one.\nKeep this exact newline.','Configuration two.']
    for index,text in enumerate(texts):
        save_json(template,[dict(note1=text)])
        guidance=load_shopping_prompt(template)
        _,record,_=shopping.webshop_run_rap(str(index),'','','Fixture initial context',
            FixtureEnvironment(),client,'fixture model','minimax',None,[],{},1,
            benign=True,test=True,to_print=False,honest_reward=True,auxiliary_prompt=guidance['text'])
        assert record['PolicyTrace'][0]['AuxiliaryPrompt']==text
        messages=requests[index]['messages']
        assert messages[1]['content'].startswith('Additional task guidance:\n'+text+'\n\n')
    assert requests[0]['messages'][0]==requests[1]['messages'][0]


@pytest.mark.parametrize('field,value',[('source_rank',2),('source_score',0.9)])
def test_independent_recheck_rejects_stale_source_summary_even_if_full_ranking_matches(tmp_path,monkeypatch,field,value):
    from src.experiments import shampoo_ablation_evaluation as evaluation
    queries=[dict(task_id=str(i),source_rank=1,source_score=0.8,
        ranking=[dict(Id='source',rank=1,score=0.8,image_sha256='image')]) for i in range(20)]
    actual=[dict(variant='clean',encoder={'fixture':True},bank_sha256='bank',queries=queries)]
    stored=copy.deepcopy(actual);stored[0]['queries'][0][field]=value
    save_json(tmp_path/'evaluation/PLAN.json',dict(fixture=True))
    save_json(tmp_path/'evaluation/RANKINGS.json',dict(rows=stored,plan_sha256=sha256_file(tmp_path/'evaluation/PLAN.json')))
    monkeypatch.setattr(evaluation,'compute_rankings',lambda root:copy.deepcopy(actual))
    with pytest.raises(ValueError,match='源商品名次或分数汇总'):
        evaluation.verify(tmp_path)


def test_independent_recheck_requires_similarity_for_every_frozen_arm(tmp_path,monkeypatch):
    from src.experiments import shampoo_ablation_evaluation as evaluation
    queries=[dict(task_id=str(i),source_rank=1,source_score=0.8,
        ranking=[dict(Id='source',rank=1,score=0.8,image_sha256='image')]) for i in range(20)]
    actual=[dict(variant='clean',encoder={'fixture':True},bank_sha256='bank',queries=queries)]
    save_json(tmp_path/'evaluation/PLAN.json',dict(fixture=True))
    save_json(tmp_path/'evaluation/RANKINGS.json',dict(rows=actual,
        plan_sha256=sha256_file(tmp_path/'evaluation/PLAN.json')))
    save_json(tmp_path/'evaluation/SIMILARITIES.json',[])
    monkeypatch.setattr(evaluation,'compute_rankings',lambda root:copy.deepcopy(actual))
    monkeypatch.setattr(evaluation,'compute_similarities',lambda root:[dict(variant='clean')])
    with pytest.raises(ValueError,match='图像相似度复核条件缺失'):
        evaluation.verify(tmp_path)


def test_perlage_no_pixel_independent_method_identity(products, tmp_path):
    from src.experiments.pipeline import generation_definition
    from src.visual.optimization import METHODS, normalize_method
    from src.visual._perlage.evidence import NO_PIXEL_METHOD, NO_PIXEL_DEFINITION, validate_complete
    from src.visual._perlage.pipeline import validate_definition, initialization_config

    assert 'PerLAGE_no_pixel' in METHODS
    assert normalize_method('perlage_no_pixel') == NO_PIXEL_METHOD

    definition = generation_definition(dict(source=products[1], target=products[0]), {}, method='PerLAGE_no_pixel')
    assert definition['method'] == NO_PIXEL_METHOD
    assert definition['definition'] == NO_PIXEL_DEFINITION
    assert definition['pixel_optimization'] is False

    cfg = validate_definition(definition)
    assert cfg.pixel_optimization is False and cfg.latent['auxiliary_levels'] == 0
    init_cfg = initialization_config(definition)
    assert init_cfg.pixel_clean_steps == init_cfg.pixel_diffusion_steps == init_cfg.pixel_levels == 0

    from src.tests.test_completion_and_order import complete_fixture
    ready = complete_fixture(products, tmp_path)
    ready['method'] = NO_PIXEL_METHOD
    ready['definition'] = NO_PIXEL_DEFINITION
    ready['pixel_optimization'] = False
    ready['auxiliary_max_levels'] = 0
    validate_complete(ready, products[1]['image'], method=NO_PIXEL_METHOD)

