from pathlib import Path
import pytest

from src.vision.audit import save_json, read_json, digest
from src.vision.image_utils import sha256_file
from src.vision._perlage.evidence import METHOD, DEFINITION, validate_complete
from src.vision._natural.evidence import STAGES
from src.vision.retrieval_objective import TRAINING_DATA_POLICY


def complete_fixture(products, tmp_path):
    proof = tmp_path/'proof.json'
    save_json(proof, {'public_binding': {'shadow_pool_used': False}})
    optimizer = tmp_path/'optimizer.json'; save_json(optimizer,{})
    initialized = tmp_path/'ready.json'; save_json(initialized,{})
    stages = {name: dict(status='completed', files={str(proof):sha256_file(proof)}) for name in STAGES}
    stages['latent_generation_refinement'].update(steps_completed=60,steps_requested=60)
    return dict(method=METHOD,definition=DEFINITION,status='completed',color_scheme='statistics',
                shadow_pool_used=False,purchase_history_used=False,victim_memory_used=False,formal_queries_used=False,
                training_data_policy=TRAINING_DATA_POLICY,stages=stages,
                initialization_identity=str(proof),initialization_optimizer_identity=str(optimizer),
                initialization_evidence={str(p):sha256_file(p) for p in (proof,optimizer,initialized)},
                candidates={'final':dict(image=products[1]['image'],sha256=products[1]['image_sha256'])})


@pytest.mark.parametrize('stage',STAGES)
def test_missing_any_stage_cannot_be_completed(products,tmp_path,stage):
    ready=complete_fixture(products,tmp_path)
    del ready['stages'][stage]
    with pytest.raises(ValueError):validate_complete(ready)


def test_initialization_alone_and_short_schedule_rejected(products,tmp_path):
    ready=complete_fixture(products,tmp_path)
    validate_complete(ready,products[1]['image'])
    ready['stages']['latent_generation_refinement']['steps_completed']=59
    with pytest.raises(ValueError):validate_complete(ready)
    ready=complete_fixture(products,tmp_path)
    del ready['initialization_evidence']
    with pytest.raises(ValueError):validate_complete(ready)


def test_both_methods_use_same_public_selection_before_generation(products,tmp_path,monkeypatch):
    import numpy as np
    from src.experiments import pipeline
    from src.vision import optimization
    events=[]
    public=tmp_path/'data/public';public.mkdir(parents=True)
    # 此夹具仅隔离编排行为，真实模型另做本地加载和推理检查。
    target=dict(products[0]);source=dict(products[1])
    for p in (target,source):p.pop('image_sha256')
    for p in (target,source):
        dest=public/(p['id']+'.png');dest.write_bytes(Path(p['image']).read_bytes());p['image']=dest.name
    save_json(public/'catalog.json',dict(schema_version=1,target=target,candidates=[source]))
    monkeypatch.setattr(pipeline,'DATA',tmp_path/'data')
    class Backend:
        snapshot={'weights':'test_fixture'}
        def __init__(self,*args):events.append('select')
        def features(self,p):
            text=np.array([1.,0.]) if p['id']==target['id'] else np.array([.72,np.sqrt(1-.72**2)])
            return dict(cls=[1.,0.],patches=[[1.,0.],[0.,1.]],text=text,clip_image=[1.,0.],clip_text=[1.,0.],aspect=1.)
        def recolor_cost(self,*args):return .01
        def close(self):pass
    config=dict(public_catalog=str(public/'catalog.json'),generation={'lucid':{'steps':2},'PerLAGE':{}})
    root=tmp_path/'run'
    selected=pipeline.select_pair(config,root,device='cpu',backend_factory=Backend)
    selected_again=pipeline.select_pair(config,root,device='cpu',backend_factory=Backend)
    assert selected_again==selected and events==['select']
    monkeypatch.setattr(pipeline,'source_snapshot',lambda:{'test':'fixture'})
    def fake_generate(source,target,output,**kwargs):
        events.append(kwargs['method'])
        assert Path(source).stem=='source' and Path(target).stem=='target'
        out=Path(output);out.mkdir(parents=True)
        image=out/'adversarial.png';image.write_bytes(Path(source).read_bytes())
        common=dict(method=kwargs['method'],status='success',output_image=str(image),output_image_sha256=sha256_file(image),
                    source_image_sha256=sha256_file(source),target_image_sha256=sha256_file(target))
        if kwargs['method']=='lucid':common['attack']=dict(steps_completed=2,steps_requested=2)
        else:
            ready=complete_fixture(products,out)
            ready['source_sha256']=sha256_file(source);ready['target_sha256']=sha256_file(target)
            ready['candidates']['final']=dict(image=str(image),sha256=sha256_file(image))
            common.update(perlage_ready=ready,shadow_pool_used=False)
        return image,common
    a=pipeline.generate(config,root,selected,'lucid',device='cpu',generator=fake_generate)
    b=pipeline.generate(config,root,selected,'PerLAGE',device='cpu',generator=fake_generate)
    assert events==['select','lucid','PerLAGE']
    assert a['source']==b['source'] and a['target']==b['target']
    assert a['shared_pool_scope']==b['shared_pool_scope']=='shared'
    assert pipeline.load_frozen(root/'methods/lucid')==a
    assert pipeline.load_frozen(root/'methods/PerLAGE')==b
    assert (root/'methods/PerLAGE/QUERY_IMAGE_REQUEST.json').is_file()
    changed=dict(config,generation={'lucid':{'steps':3}})
    with pytest.raises(ValueError):pipeline.generate(changed,root,selected,'lucid',generator=fake_generate)


def test_shopping_is_disabled_before_clients_or_environment_are_loaded(tmp_path):
    from src.experiments.shopping_evaluation import run_shopping_evaluation
    with pytest.raises(PermissionError):
        run_shopping_evaluation(tmp_path/'methods/lucid',model_config=tmp_path/'missing.json')
