import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from src.visual.pair_selection import LocalSelectionBackend, unit_rows, uniform_draw
from src.visual.audit import read_json
from src.visual.image_utils import sha256_file


def test_review_package_integrity():
    base = Path('output/public_pair_selection/groups9_review_20261108')
    assert base.is_dir(), 'Review package base directory does not exist'
    
    # 1. Check overview files
    manifest_file = base / 'REVIEW_MANIFEST.json'
    assert manifest_file.is_file(), 'REVIEW_MANIFEST.json missing'
    overview_file = base / 'OVERVIEW.md'
    assert overview_file.is_file(), 'OVERVIEW.md missing'
    contact_all = base / 'CONTACT_SHEET_ALL9.jpg'
    assert contact_all.is_file(), 'CONTACT_SHEET_ALL9.jpg missing'
    
    with Image.open(contact_all) as img:
        assert img.width > 2000 and img.height > 1500

    manifest = read_json(manifest_file)
    assert len(manifest['groups']) == 9
    
    # 2. Check each group
    expected_groups = [
        ('01_camera', 'B08V134QWR', 'B071ZFHTZN'),
        ('02_chair', 'B09CYFD5BX', 'B08GCSRXKS'),
        ('03_cookies', 'B08Z8MHWVS', 'B07BBLWTLM'),
        ('04_curtain', 'B086PW7TWX', 'B09PHLS2QD'),
        ('05_hair_extension', 'B01460C2I2', 'B07Z7MQSM1'),
        ('06_jeans', 'B09HH96NJY', 'B07FRKQNFV'),
        ('07_shampoo', 'B00BDJODWI', 'B08XYBS678'),
        ('08_speaker', 'B01N47U1VI', 'B01590FHS0'),
        ('09_toothbrush', 'B09J81RMLD', 'B09PNNXYJ4'),
    ]
    
    for key, expected_tid, expected_sid in expected_groups:
        gdir = base / 'groups' / key
        assert gdir.is_dir(), f'Group directory missing: {key}'
        defn_file = gdir / 'definition.json'
        assert defn_file.is_file(), f'definition.json missing: {key}'
        defn = read_json(defn_file)
        
        assert defn['group'] == key
        assert defn['target']['id'] == expected_tid
        assert defn['source']['id'] == expected_sid
        
        # Verify files exist and are readable images
        for fname in ('target.jpg', 'source.jpg', 'target_foreground.png', 'source_foreground.png', 'contact_sheet.jpg'):
            fpath = gdir / fname
            assert fpath.is_file(), f'Missing {fname} in {key}'
            with Image.open(fpath) as im:
                im.verify()


def test_dino_shape_similarities():
    base = Path('output/public_pair_selection/groups9_review_20261108')
    feat_cache = base / 'test_verification_features'
    backend = LocalSelectionBackend({}, feat_cache, 'cuda')
    try:
        groups_dir = base / 'groups'
        for g in sorted(groups_dir.iterdir()):
            if not g.is_dir():
                continue
            defn = read_json(g / 'definition.json')
            t_img_path = g / 'target.jpg'
            s_img_path = g / 'source.jpg'
            t_feat = backend.features({'id': defn['target']['id'], 'image': str(t_img_path), 'image_sha256': sha256_file(t_img_path)})
            s_feat = backend.features({'id': defn['source']['id'], 'image': str(s_img_path), 'image_sha256': sha256_file(s_img_path)})
            sim = float(np.clip(unit_rows(s_feat['shape']) @ unit_rows(t_feat['shape']), -1, 1))
            diff = abs(defn['shape_similarity'] - sim)
            print(f"{defn['group']}: reported={defn['shape_similarity']:.4f}, computed={sim:.4f}, diff={diff:.6f}")
            assert diff < 0.005, f"Similarity mismatch in {defn['group']}: reported {defn['shape_similarity']}, computed {sim}"
    finally:
        backend.close()
