from src.paths import ProjectPath as Path
import copy
import pytest
from PIL import Image
from src.visual.audit import save_json,read_json,digest
from src.visual.image_utils import sha256_file
from src.experiments import public_pair_selection as flow
from src.visual.pair_assets import load_pair


def test_batch_draw_annotates_only_chosen_pair_then_publishes_named_assets(products,tmp_path,monkeypatch,calibration_factory,public_feature_stub):
    public=tmp_path/'data/public';public.mkdir(parents=True)
    root=tmp_path/'output/public_pair_selection/simple_v2';working=public/'selection_runs/simple_v2';working.mkdir(parents=True)
    monkeypatch.setattr(flow,'PUBLIC',public);monkeypatch.setattr(flow,'OUTPUT',tmp_path/'output')
    cfg=calibration_factory(directory=public/'calibration')
    config=dict(schema_version=2,seed=42,candidate_limit=32,calibration_path=cfg.calibration_path,groups=[dict(key='test_chair',target_id='A000000001')])
    short_products=[];downloads={}
    for index,p in enumerate(products):
        image=working/(str(index)+'.png')
        with Image.open(p['image']) as opened:opened.resize((80,80)).save(image)
        pid=f'A{index+1:09d}'
        record=dict(id=pid,title='Public salon chair '+str(index),image_url='https://m.media-amazon.com/'+pid+'.jpg',source_url='https://www.amazon.com/dp/'+pid)
        short_products.append(record);downloads[pid]=dict(record,image=str(image),image_sha256=sha256_file(image))
    short=dict(schema_version=2,groups=[dict(key='test_chair',target=short_products[0],candidates=short_products[1:])])
    save_json(working/'candidate_shortlist.json',short)
    save_json(working/'download_manifest.json',dict(shortlist_sha256=digest(short),records=downloads,failures={}))
    flow.freeze_plan(config,root)
    class Backend:
        snapshot={'fixture':True}
        def __init__(self,*args):pass
        def features(self,p):return {'shape':[1.,0.]}
        def close(self):pass
    bundle=flow.select_groups(config,root,working,device='cpu',backend_factory=Backend)
    chosen=bundle['groups'][0]['source']['id']
    assert chosen in {p['id'] for p in short_products[1:]}
    selected,subset=flow.selected_inputs(root,working)
    assert set(subset['records'])=={chosen,'A000000001'} and len(downloads)==3
    import src.visual.public_annotations as annotation
    def propose(downloads,output):
        assert len(downloads['records'])==2
        proposals={}
        for pid,p in downloads['records'].items():
            mask=Path(output)/(pid+'.png');mask.parent.mkdir(parents=True,exist_ok=True);Image.new('L',(80,80),255).save(mask)
            proposals[pid]=dict(boxes=[[0,0,80,80]],foreground_mask=str(mask),text_regions=[])
        document=dict(proposals=proposals);save_json(Path(output)/'annotation_manifest.json',document);return document
    monkeypatch.setattr(annotation,'annotate_downloads',propose)
    monkeypatch.setattr(annotation,'contact_sheets',lambda *a,**k:None)
    proposed=flow.annotate_selected(root,working)
    save_json(working/'selected_visual_reviews.json',dict(reviewer='合成测试，非实际审核',annotation_manifest_sha256=digest(proposed),
        records={pid:dict(accepted=True,image_sha256=downloads[pid]['image_sha256'],proposal_sha256=digest(proposal)) for pid,proposal in proposed['proposals'].items()}))
    result=flow.publish_groups(root,working)
    assert len(result['groups'])==1
    directory=root/'data/groups/test_chair'
    pair=load_pair(directory/'pair.json',root/'data')
    assert Path(pair['source']['image']).name=='source.png' and Path(pair['target']['image']).name=='target.png'
    assert Path(pair['source']['image']).read_bytes()==Path(downloads[chosen]['image']).read_bytes()
    generated=read_json(directory/'pipeline.json')
    from src.experiments.pipeline import load_config, generation_definition
    assert load_config(directory/'pipeline.json') == generated
    for method in ('PerLAGE', 'PerLAGE_no_pixel'):
        definition = generation_definition(pair, generated['generation'][method], method)
        assert definition['data']['training_queries'] == [generated['generation']['lucid']['target_text']]
    assert generated['public_pair']==str(directory/'pair.json')
    assert generated['evaluation']['shared_pool']==str(flow.SHARED_MEMORY)
    assert 'simple_v2' in generated['evaluation']['query_manifest']
    assert flow.publish_groups(root,working)==result
    config_path=public/'config.json';save_json(config_path,config)
    save_json(root/'RUN_STATUS.json',dict(status='published'))
    assert flow.main(['report','--config',str(config_path),'--root',str(root),'--public-root',str(working)])==0
    assert read_json(root/'RUN_STATUS.json')['status']=='published'


def test_raw_pipeline_stops_after_draw_when_only_selected_annotations_are_missing(products,tmp_path,monkeypatch,calibration_factory,public_feature_stub):
    from dataclasses import asdict
    from src.experiments import pipeline
    public=tmp_path/'data/public';public.mkdir(parents=True)
    monkeypatch.setattr(pipeline,'DATA',public.parent)
    records=[]
    for p in products[:2]:
        p=copy.deepcopy(p);dest=public/Path(p['image']).name;dest.write_bytes(Path(p['image']).read_bytes());p['image']=str(dest)
        for k in ('boxes','text_regions','text_regions_reviewed'):p.pop(k)
        records.append(p)
    catalog=public/'catalog.json';save_json(catalog,dict(schema_version=2,target=records[0],candidates=[records[1]]))
    cfg=calibration_factory(directory=public/'calibration')
    class Backend:
        snapshot={'fixture':True}
        def __init__(self,*args):pass
        def features(self,p):return {'shape':[1.,0.]}
        def close(self):pass
    root=tmp_path/'run'
    with pytest.raises(pipeline.SelectedAnnotationsPending):
        pipeline.select_pair(dict(public_catalog=str(catalog),selection=asdict(cfg)),root,backend_factory=Backend)
    assert (root/'selection/DRAWN_PAIR.json').is_file() and not (root/'selection/SELECTED_PAIR.json').exists()
    before=sha256_file(root/'selection/DRAWN_PAIR.json')
    for p in records:p.update(boxes=[[0,0,24,24]],text_regions=[],text_regions_reviewed=True)
    save_json(catalog,dict(schema_version=2,target=records[0],candidates=[records[1]]))
    selected=pipeline.select_pair(dict(public_catalog=str(catalog),selection=asdict(cfg)),root,backend_factory=Backend)
    assert selected['source']['id']==records[1]['id'] and selected['annotation_binding']['drawn_pair_sha256']==before
    assert sha256_file(root/'selection/DRAWN_PAIR.json')==before
