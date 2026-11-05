import copy
from dataclasses import replace
import numpy as np
from PIL import Image
import pytest
from src.vision.audit import read_json,save_json,digest
from src.vision.pair_selection import SelectionConfig,choose_pair,validate_product,verify_selection,uniform_draw
from src.vision.product_types import type_compatibility,classify_public_type
from src.vision.shape_calibration import fit_threshold,load_calibration
from src.vision.shape_features import shape_input


class ShapeBackend:
    snapshot={'fixture':True}
    def features(self,product):return {'shape':[1.,0.]}
    def close(self):pass


def test_uniform_selection_is_order_independent_and_not_maximum_score(products,calibration_factory,tmp_path):
    cfg=calibration_factory();target=products[0];candidates=[]
    for i in range(12):
        p=dict(products[1],id=f'candidate{i:02d}',title=f'Shampoo public product {i}')
        image=tmp_path/f'candidate{i}.png';Image.new('RGB',(24,24),(40+i*7,80+i*5,100+i*2)).save(image)
        p.update(image=str(image),image_sha256=__import__('hashlib').sha256(image.read_bytes()).hexdigest());candidates.append(p)
    class Backend(ShapeBackend):
        def features(self,p):
            score=1. if p['id']==target['id'] else .6+.025*int(p['id'][-2:])
            return {'shape':[score,np.sqrt(1-score**2)]}
    result=choose_pair(target,candidates,Backend(),cfg)
    again=choose_pair(target,list(reversed(candidates)),Backend(),cfg)
    assert result==again
    assert result['selected_source_id']!='candidate11'
    result.update(source=next(p for p in candidates if p['id']==result['selected_source_id']),target=target)
    verify_selection(result)
    result['draw']['index']=(result['draw']['index']+1)%12
    with pytest.raises(ValueError,match='抽样记录'):verify_selection(result)


@pytest.mark.parametrize('mutation,reason',[
    ({'id':'product0'},'same_product_id'),({'title':'PUBLIC SHAMPOO PRODUCT NUMBER 0!!!'},'same_normalized_title'),
    ({'owned':False},'source_not_attacker_owned'),({'title':'Blue public jeans'},'obvious_product_type_conflict'),
    ({'title':'A public mystery object'},'public_type_review_required')])
def test_explicit_gates(products,calibration_factory,mutation,reason):
    p=dict(products[1],**mutation)
    result=choose_pair(products[0],[p],ShapeBackend(),calibration_factory())
    assert result['status']=='no_eligible_pair' and reason in result['candidates'][0]['rejected_reasons']


def test_identical_images_and_duplicate_candidates_are_excluded(products,calibration_factory):
    cfg=calibration_factory()
    same=dict(products[1],image=products[0]['image'],image_sha256=products[0]['image_sha256'])
    duplicate=dict(products[1],id='z_duplicate',title='Public other shampoo')
    r=choose_pair(products[0],[same,duplicate],ShapeBackend(),cfg)
    assert 'same_image_bytes' in r['candidates'][0]['rejected_reasons']
    # 第二条图是正常候选；重复候选只保留确定的一个代表。
    r=choose_pair(products[0],[products[1],duplicate],ShapeBackend(),cfg)
    assert sum(x['eligible'] for x in r['candidates'])==1


def test_threshold_missing_and_legacy_parameters_are_not_silently_accepted():
    with pytest.raises(ValueError,match='真实公开开发集'):SelectionConfig().validate()
    with pytest.raises(TypeError):SelectionConfig(shape_min=.65)
    with pytest.raises(ValueError):SelectionConfig(seed=True,calibration_path='x').validate()


def test_calibration_fit_endpoints_and_tie_rule():
    result=fit_threshold([.8,.9,.1,.2],[1,1,0,0])
    assert result['threshold']==pytest.approx(.5) and result['error_count']==0
    tied=fit_threshold([.5,.5],[1,0])
    assert tied['threshold']==-1 and tied['tie_rule']=='smallest_threshold'
    for scores,labels in [([float('nan'),.2],[1,0]),([.8,.1,.2],[1,0,0]),([.8,.2],[True,False])]:
        with pytest.raises(ValueError):fit_threshold(scores,labels)


def test_calibration_identity_scores_and_development_separation(products,calibration_factory):
    cfg=calibration_factory()
    with pytest.raises(ValueError,match='模型或预处理'):load_calibration(cfg.calibration_path,{'other_model':True})
    dev=dict(products[1],id='development0')
    with pytest.raises(ValueError,match='开发集商品重叠'):choose_pair(products[0],[dev],ShapeBackend(),cfg)
    proof=read_json(cfg.calibration_path);proof['fit']['threshold']=.99
    proof['certificate_sha256']=digest({k:v for k,v in proof.items() if k!='certificate_sha256'})
    save_json(cfg.calibration_path,proof)
    with pytest.raises(ValueError,match='逐对开发分数'):load_calibration(cfg.calibration_path,ShapeBackend.snapshot)


def test_new_sampler_stops_empty_and_honors_reserved_ids(products,calibration_factory):
    cfg=calibration_factory()
    r=choose_pair(products[0],products[1:],ShapeBackend(),cfg,excluded_source_ids=[p['id'] for p in products[1:]])
    assert r['selected_source_id'] is None and r['draw'] is None
    assert all('reserved_in_public_batch' in x['rejected_reasons'] for x in r['candidates'])


def test_type_gate_is_coarse_and_public_review_is_bound_to_title():
    assert type_compatibility({'title':'Hydraulic barber chair'},{'title':'Office chair'})['compatible']
    assert type_compatibility({'title':'Hair shampoo'},{'title':'Hair conditioner'})['compatible']
    assert not type_compatibility({'title':'Kids toothbrush'},{'title':'Bluetooth speaker'})['compatible']
    p=dict(title='Public ambiguous object',type_review=dict(category='oral_care',reviewer='公开核对夹具',basis='合成测试',title_sha256=digest('Public ambiguous object')))
    assert classify_public_type(p)['category']=='oral_care'
    with pytest.raises(ValueError):classify_public_type(dict(p,title='Changed public title'))


def test_grayscale_letterbox_keeps_geometry_and_equal_channels(tmp_path):
    path=tmp_path/'wide.png';Image.new('RGB',(200,100),(220,30,60)).save(path)
    prepared=np.asarray(shape_input(path))
    assert prepared.shape==(224,224,3) and np.all(prepared[:,:,0]==prepared[:,:,1]) and np.all(prepared[:,:,1]==prepared[:,:,2])
    assert np.all(prepared[0]==127) and np.any(prepared[100]!=127)


def test_only_selected_products_require_generation_annotations(products,tmp_path):
    p={k:v for k,v in products[1].items() if k not in ('boxes','text_regions','text_regions_reviewed')}
    assert validate_product(p,tmp_path,owned=True,generation_ready=False)['id']==p['id']
    with pytest.raises(ValueError,match='核对'):validate_product(p,tmp_path,owned=True)
    with pytest.raises(ValueError,match='未知字段'):validate_product(dict(p,source_history='私有历史'),tmp_path,generation_ready=False)


def test_public_path_boundary_is_checked_before_any_file_content_read(products,tmp_path,monkeypatch):
    import src.vision.pair_selection as module
    public=tmp_path/'public';public.mkdir()
    secret=tmp_path/'secret.dat';secret.write_bytes(b'private payload')
    def forbidden(*args):raise AssertionError('不得先读取或计算私有文件摘要')
    monkeypatch.setattr(module,'sha256_file',forbidden)
    p=dict(products[1],image=str(secret))
    with pytest.raises(ValueError,match='读取内容前'):validate_product(p,tmp_path,generation_ready=False,public_root=public)
