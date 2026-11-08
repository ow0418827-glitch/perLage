import copy
import io
from src.paths import ProjectPath as Path

import numpy as np
from PIL import Image,ImageDraw
import pytest

from src.visual.selection.webshop_public import (public_projection,simulated_controlled,
    extract_public_candidates,random_shortlist,download_public_image)
from src.visual.audit import save_json,digest
from src.visual.public_annotations import (object_prior,remove_connected_background,regions_from_ocr,apply_reviews)
from src.experiments.public_pair_selection import validate_config,pipeline_config


def public_record(pid='A000000001',name='Public salon chair'):
    return dict(asin=pid,name=name,images=['https://m.media-amazon.com/images/I/test.jpg'])


def test_projection_ignores_environment_and_private_like_fields():
    original=public_record()
    changed=dict(original,category_tree={'private':'tree'},instructions=['future task'],
                 price=999,history=['purchase'],reward=1,attributes=['internal'])
    assert public_projection(original)==public_projection(changed)
    assert set(public_projection(changed))=={'id','title','image_url','source_url'}
    assert public_projection(dict(original,images=['http://localhost/private'])) is None


def test_simulated_control_is_stable_and_always_excludes_targets():
    targets={'A000000001'}
    assert not simulated_controlled('A000000001',42,targets)
    first=[simulated_controlled(f'{i:010d}',42,targets) for i in range(30)]
    assert first==[simulated_controlled(f'{i:010d}',42,targets) for i in range(30)]
    assert any(first) and not all(first)


def test_raw_dataset_preparation_keeps_only_public_matching_candidates(tmp_path):
    target='A000000001'
    sources=[f'{i:010d}' for i in range(100) if simulated_controlled(f'{i:010d}',42,{target})][:4]
    rows=[public_record(target)]+[public_record(pid) for pid in sources]+[public_record('B000000001','Unrelated public item')]
    path=tmp_path/'products.json';save_json(path,rows)
    groups=[dict(key='chair',target_id=target)]
    result=extract_public_candidates(path,groups,tmp_path/'public',lexical_limit=3)
    assert result['scanned_products']==6
    assert len(result['groups'][0]['candidates'])==3
    assert result['groups'][0]['target']['id']==target
    assert all(p['id']!=target for p in result['groups'][0]['candidates'])
    assert extract_public_candidates(path,groups,tmp_path/'public',lexical_limit=3)==result


def test_public_index_reuses_scan_across_runs_and_binds_source(tmp_path,monkeypatch):
    import ijson
    from src.visual.selection.webshop_public import build_public_index
    path=tmp_path/'products.json';save_json(path,[public_record()])
    index=tmp_path/'public.sqlite';first=build_public_index(path,index)
    monkeypatch.setattr(ijson,'items',lambda *a,**k:(_ for _ in ()).throw(AssertionError('不能重复扫描原始文件')))
    assert build_public_index(path,index)==first
    assert build_public_index(path,index,verify_source=True)==first
    path.write_text('[]',encoding='utf-8')
    with pytest.raises(ValueError,match='源文件'):build_public_index(path,index)


def test_development_items_are_excluded_before_candidate_download(tmp_path):
    target='A000000001';ids=[f'{i:010d}' for i in range(100) if simulated_controlled(f'{i:010d}',42,{target})][:5]
    path=tmp_path/'products.json';save_json(path,[public_record(target)]+[public_record(x) for x in ids])
    result=extract_public_candidates(path,[dict(key='chair',target_id=target)],tmp_path/'public',lexical_limit=32,excluded_ids=ids[:2])
    assert {x['id'] for x in result['groups'][0]['candidates']}==set(ids[2:])
    with pytest.raises(ValueError,match='目标与开发'):extract_public_candidates(path,[dict(key='chair',target_id=target)],tmp_path/'other',excluded_ids=[target])


def test_shortlist_preserves_fixed_random_candidates_without_text_model(tmp_path):
    target=public_projection(public_record())
    candidates=[public_projection(public_record(f'{i:010d}',f'Public source {i}')) for i in range(3)]
    extracted=dict(groups=[dict(key='chair',target=target,candidates=candidates,counts={})])
    out=random_shortlist(extracted,tmp_path)
    assert out['groups'][0]['candidates']==candidates and out['text_model_used'] is False


def test_download_uses_current_cache_and_rejects_digest_changes(tmp_path):
    stream=io.BytesIO();Image.new('RGB',(80,80),'blue').save(stream,format='PNG')
    calls=[]
    def fetch(url):calls.append(url);return stream.getvalue()
    record=public_projection(public_record())
    first=download_public_image(record,tmp_path,fetcher=fetch)
    assert download_public_image(record,tmp_path,fetcher=fetch)==first and len(calls)==1
    Path(first['image']).write_bytes(b'changed')
    with pytest.raises(ValueError):download_public_image(record,tmp_path,fetcher=fetch)


def test_background_cleanup_preserves_enclosed_white_object_interior():
    image=Image.new('RGB',(100,100),'white');draw=ImageDraw.Draw(image)
    draw.rectangle((20,20,80,80),outline='black',width=3)
    mask=remove_connected_background(image,np.ones((100,100),dtype=bool))
    assert not mask[0,0] and mask[50,50] and mask[20,20]
    rough,box,point=object_prior(image)
    assert rough[point[1],point[0]] and box[0]<=20 and box[2]>=80


def test_ocr_boxes_are_padded_and_normalized_without_changing_image():
    result=[([[10,10],[30,10],[30,20],[10,20]],'label',.9)]
    regions,texts=regions_from_ocr(result,100,100)
    assert regions==[[.07,.07,.33,.23]]
    assert texts[0]['text']=='label'


def test_review_cannot_be_fabricated_by_marking_a_single_global_flag(tmp_path):
    p=public_projection(public_record());p['image']='test.png'
    shortlist=dict(groups=[dict(key='chair',target=p,candidates=[])])
    annotations=dict(proposals={p['id']:dict(text_regions=[],foreground_mask='m.png',boxes=[[0,0,1,1]])})
    downloads=dict(records={p['id']:dict(image_sha256='f'*64)})
    review=dict(reviewer='actual reviewer',annotation_manifest_sha256=digest(annotations),records={})
    with pytest.raises(ValueError):apply_reviews(shortlist,downloads,annotations,review,tmp_path)


def test_multiple_targets_keep_one_shared_pool_and_separate_query_images(tmp_path):
    a=pipeline_config('02_chair',tmp_path/'chair.json',tmp_path)
    b=pipeline_config('09_toothbrush',tmp_path/'brush.json',tmp_path)
    assert a['evaluation']['shared_pool']==b['evaluation']['shared_pool']
    assert a['evaluation']['query_manifest']!=b['evaluation']['query_manifest']
    assert a['run_dir']!=b['run_dir']


def test_batch_config_rejects_path_traversal():
    config=dict(schema_version=2,seed=42,candidate_limit=32,calibration_path='public_certificate.json',
        groups=[dict(key='../escape',target_id='A000000001')])
    with pytest.raises(ValueError):validate_config(config)


def test_foreground_mask_changes_are_bound_to_core_input_identity(products,tmp_path):
    from src.visual.pair_selection import validate_product
    p=dict(products[1]);p.pop('image_sha256')
    mask=tmp_path/'mask.png';Image.new('L',(24,24),255).save(mask);p['foreground_mask']=str(mask)
    first=validate_product(p,tmp_path,owned=True)
    changed=Image.new('L',(24,24),255);changed.putpixel((0,0),0);changed.save(mask)
    second=validate_product(p,tmp_path,owned=True)
    assert first['foreground_mask_sha256']!=second['foreground_mask_sha256']
