import copy
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from src.vision.audit import read_json, save_json
from src.vision.image_utils import sha256_file
from src.vision.pair_assets import load_pair, publish_pair
from src.vision.pair_selection import choose_pair, validate_product


@pytest.fixture
def selected_public_pair(products, tmp_path,calibration_factory):
    public = tmp_path / 'data/public'
    cache = public / 'cache'
    cache.mkdir(parents=True)
    records = []
    for index, product in enumerate(products[:2]):
        record = copy.deepcopy(product)
        record.pop('image_sha256')
        destination = cache / ('target.png' if index == 0 else 'source.jpg')
        with Image.open(record['image']) as image:
            image.save(destination)
        mask = cache / f'mask{index}.png'
        Image.new('L', (24, 24), 255).save(mask)
        record.update(image=str(destination), foreground_mask=str(mask))
        records.append(validate_product(record, cache, owned=index == 1))
    target, source = records

    class Backend:
        snapshot = {'fixture': True}
        def features(self, product):
            return dict(shape=[1.,0.])

    selection = choose_pair(target, [source], Backend(),calibration_factory(directory=public/'calibration'))
    selection.update(source=source, target=target, binding={'models_configuration': {}})
    selected_path = tmp_path / 'run/selection/SELECTED_PAIR.json'
    save_json(selected_path, selection)
    return public, cache, selected_path, public / 'groups/02_chair'


def test_named_pair_preserves_original_bytes_and_frozen_choice(selected_public_pair):
    public, _, selected_path, directory = selected_public_pair
    original = read_json(selected_path)
    evidence_hash = sha256_file(selected_path)
    selected = publish_pair(selected_path, directory, public)
    assert Path(selected['source']['image']) == directory / 'source.jpg'
    assert Path(selected['target']['image']) == directory / 'target.png'
    for role in ('source', 'target'):
        assert Path(selected[role]['image']).read_bytes() == Path(original[role]['image']).read_bytes()
        assert selected[role]['id'] == original[role]['id']
        assert selected[role]['title'] == original[role]['title']
    assert selected['candidates'] == original['candidates']
    assert sha256_file(selected_path) == evidence_hash
    assert publish_pair(selected_path, directory, public) == selected


def test_pipeline_reads_group_folder_after_download_cache_is_removed(selected_public_pair, tmp_path, monkeypatch):
    from src.experiments import pipeline
    public, cache, selected_path, directory = selected_public_pair
    selected = publish_pair(selected_path, directory, public)
    for file in cache.iterdir():file.unlink()
    cache.rmdir()
    monkeypatch.setattr(pipeline, 'DATA', public.parent)
    config = dict(public_pair=str(directory / 'pair.json'))

    def forbidden_backend(*args, **kwargs):
        raise AssertionError('已确定图对不应重新筛选或加载候选图片')

    actual = pipeline.select_pair(config, tmp_path / 'method_run', device='cpu', backend_factory=forbidden_backend)
    assert actual['source'] == selected['source'] and actual['target'] == selected['target']
    definition = pipeline.generation_definition(actual, {})
    assert definition['data']['source'] == str(directory / 'source.jpg')
    assert definition['data']['target'] == str(directory / 'target.png')


def test_named_image_change_is_rejected_before_generation(selected_public_pair):
    public, _, selected_path, directory = selected_public_pair
    selected = publish_pair(selected_path, directory, public)
    Path(selected['source']['image']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='冻结图片'):
        load_pair(directory / 'pair.json', public)


def test_another_frozen_choice_cannot_overwrite_the_same_group(selected_public_pair):
    public, _, selected_path, directory = selected_public_pair
    selected = publish_pair(selected_path, directory, public)
    original = read_json(selected_path)
    original['test_note'] = 'another frozen selection'
    save_json(selected_path, original)
    with pytest.raises(ValueError, match='另一份冻结选择'):
        publish_pair(selected_path, directory, public)
    assert sha256_file(selected['source']['image']) == selected['source']['image_sha256']


def test_pair_manifest_cannot_read_evidence_outside_the_group(selected_public_pair):
    public, _, selected_path, directory = selected_public_pair
    publish_pair(selected_path, directory, public)
    manifest = read_json(directory / 'pair.json')
    manifest['selection_record'] = '../other/selection.json'
    save_json(directory / 'pair.json', manifest)
    with pytest.raises(ValueError, match='公开数据目录'):
        load_pair(directory / 'pair.json', public)


def test_config_distinguishes_candidate_selection_and_a_determined_pair(tmp_path):
    from src.experiments.pipeline import load_config
    path = tmp_path / 'config.json'
    config = dict(schema_version=1, method='PerLAGE', public_pair='pair.json')
    save_json(path, config)
    assert load_config(path) == config
    save_json(path, dict(config, public_catalog='catalog.json'))
    with pytest.raises(ValueError, match='只能使用其中一种'):
        load_config(path)
