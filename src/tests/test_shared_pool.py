import copy
from pathlib import Path
import numpy as np
import pytest

from src.memory.shared_pool import build_shared_pool, load_shared_pool, evaluation_banks
from src.vision.audit import save_json
from src.vision.image_utils import sha256_file
from src.vision.visual_retrieval import VisualMemoryRetriever


class ToyEncoder:
    snapshot = {'provider': 'test_fixture', 'weights': 'deterministic_rgb_v1'}
    def __init__(self): self.closed = False
    def encode(self, path):
        from PIL import Image
        v = np.asarray(Image.open(path).convert('RGB')).mean((0, 1)) + 1
        return v / np.linalg.norm(v)
    def encode_text(self, text):
        v = np.array([sum(text.encode('utf8')) % 17 + 1., 2., 3.])
        return v / np.linalg.norm(v)
    def encode_joint(self, text, path):
        v = self.encode(path) + self.encode_text(text)
        return v / np.linalg.norm(v)
    def close(self): self.closed = True


def test_shared_pool_merges_inputs_and_copies_images(pool_inputs, tmp_path):
    path, rows = pool_inputs
    dest = tmp_path / 'private/shared_memory.json'
    build_shared_pool([path, path], dest, kind='purchase_memory')
    entries = load_shared_pool(dest)
    assert len(entries) == len(rows)
    assert len({e['Id'] for e in entries}) == len(rows)
    for e in entries:
        assert e['Actions'] == ['完整操作与观察']
        assert (dest.parent / e['PrimaryImage']['LocalPath']).is_file()
    with pytest.raises(FileExistsError):
        build_shared_pool([path], dest, kind='purchase_memory')


def test_simulated_gallery_is_never_reported_as_purchase(pool_inputs, tmp_path):
    path, rows = pool_inputs
    with pytest.raises(ValueError):
        build_shared_pool([path], tmp_path / 'bad.json', kind='simulated_gallery')
    for e in rows: e['Completed'] = False
    save_json(path, rows)
    out = tmp_path / 'good.json'
    build_shared_pool([path], out, kind='simulated_gallery')
    assert all(not e['Completed'] for e in load_shared_pool(out))


def test_group_bank_is_rejected(pool_inputs, tmp_path):
    path, rows = pool_inputs
    save_json(path, {'scope': 'per_group', 'groups': {'one': rows}})
    with pytest.raises(ValueError): load_shared_pool(path)


def test_mixed_shared_pool_preserves_each_records_actual_kind(pool_inputs, tmp_path):
    path, rows = pool_inputs
    for row in rows:
        row['RecordKind'] = 'purchase_memory'
    rows[-1]['RecordKind'] = 'simulated_gallery'
    rows[-1]['Completed'] = False
    save_json(path, rows)
    output = tmp_path / 'mixed.json'
    build_shared_pool([path], output, kind='preserve')
    entries = load_shared_pool(output)
    assert [r['RecordKind'] for r in entries] == ['purchase_memory', 'purchase_memory', 'simulated_gallery']


def test_substitution_preserves_history_and_base_pool_and_refreshes_vectors(pool_inputs, products, tmp_path):
    path, rows = pool_inputs
    dest = tmp_path / 'shared_memory.json'
    build_shared_pool([path], dest, kind='purchase_memory')
    before = dest.read_bytes()
    entries = load_shared_pool(dest)
    banks, target_id = evaluation_banks(entries, dest, products[1], products[2]['image'])
    assert dest.read_bytes() == before
    for a, b in zip(banks['clean'], banks['modified']):
        assert {k:v for k,v in b.items() if k != 'RetrievalImage'} == a
    bank_path = tmp_path / 'bank.json'
    save_json(bank_path, banks['clean'])
    ranker = VisualMemoryRetriever(ToyEncoder(), bank_path, cache_dir=None)
    first = next(r for r in ranker.rank(banks['clean'], '模糊请求', products[0]['image']) if r['Id'] == target_id)
    save_json(bank_path, banks['modified'])
    second = next(r for r in ranker.rank(banks['modified'], '模糊请求', products[0]['image']) if r['Id'] == target_id)
    assert first['image_sha256'] != second['image_sha256']
    assert first['bank_sha256'] != second['bank_sha256']
    assert first['score'] != second['score']


def test_unrelated_history_never_changes_title_vector(pool_inputs, products, tmp_path):
    path, entries = pool_inputs
    save_json(path, entries)
    ranker = VisualMemoryRetriever(ToyEncoder(), path, cache_dir=None)
    before = ranker.rank(entries, '泛指', products[0]['image'])
    entries[0].update(Instruction='更换私有需求', Actions=['更换历史'], FinalPrice=999)
    save_json(path, entries)
    after = ranker.rank(entries, '泛指', products[0]['image'])
    assert [(r['Id'],r['score']) for r in before] == [(r['Id'],r['score']) for r in after]
