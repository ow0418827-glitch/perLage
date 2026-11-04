import numpy as np
import pytest
from PIL import Image

from src.vision.audit import save_json
from src.vision.image_utils import sha256_file


@pytest.fixture
def products(tmp_path):
    records = []
    for i, color in enumerate(((140, 80, 20), (25, 100, 160), (50, 160, 25))):
        path = tmp_path / f'product{i}.png'
        pixels = np.full((24, 24, 3), color, dtype=np.uint8)
        pixels[4:8, 4:8] = 250
        Image.fromarray(pixels).save(path)
        records.append(dict(id=f'product{i}', title=f'Public product number {i}', image=str(path),
                            image_sha256=sha256_file(path), owned=i != 0, boxes=[[0, 0, 24, 24]],
                            text_regions=[], text_regions_reviewed=True, source_note='公开输入夹具'))
    return records


@pytest.fixture
def pool_inputs(products, tmp_path):
    entries = []
    for i, product in enumerate(products):
        entries.append(dict(Id=f'original_{i}', FinalASIN=product['id'], Completed=True,
                            Instruction='原始历史需求', Actions=['完整操作与观察'], FinalPrice=12.5,
                            PrimaryImage=dict(ASIN=product['id'], Title=product['title'],
                                              LocalPath=product['image'], SHA256=product['image_sha256'])))
    path = tmp_path / 'input.json'
    save_json(path, entries)
    return path, entries
