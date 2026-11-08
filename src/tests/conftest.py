import numpy as np
from src.paths import ProjectPath as Path
import pytest
from PIL import Image

from src.visual.audit import save_json
from src.visual.image_utils import sha256_file


@pytest.fixture
def products(tmp_path):
    records = []
    for i, color in enumerate(((140, 80, 20), (25, 100, 160), (50, 160, 25))):
        path = tmp_path / f'product{i}.png'
        pixels = np.full((24, 24, 3), color, dtype=np.uint8)
        pixels[4:8, 4:8] = 250
        Image.fromarray(pixels).save(path)
        records.append(dict(id=f'product{i}', title=f'Public shampoo product number {i}', image=str(path),
                            image_sha256=sha256_file(path), owned=i != 0, boxes=[[0, 0, 24, 24]],
                            text_regions=[], text_regions_reviewed=True, source_note='公开输入夹具'))
    return records


@pytest.fixture
def calibration_factory(tmp_path):
    """真实跑拟合函数的合成单元夹具；这些标签不是生产开发集。"""
    from src.visual.shape_calibration import calibrate
    from src.visual.pair_selection import SelectionConfig
    def factory(snapshot=None,directory=None):
        directory=Path(directory or tmp_path/'public_calibration');directory.mkdir(parents=True,exist_ok=True)
        records=[]
        for i in range(4):
            image=directory/f'dev{i}.png';Image.new('RGB',(80,80),(31+i*17,40+i*11,80+i*13)).save(image)
            records.append(dict(id=f'development{i}',title=f'Public shampoo development {i}',image=str(image),source_note='合成测试夹具'))
        doc=dict(schema_version=1,reviewer='测试夹具，不作为真实视觉标签',pairs=[
            dict(left=records[0],right=records[1],label=1,review_note='合成测试正标签'),
            dict(left=records[2],right=records[3],label=0,review_note='合成测试负标签')])
        manifest=directory/'development.json';save_json(manifest,doc)
        class Backend:
            def __init__(self):self.snapshot=snapshot or {'fixture':True}
            def features(self,p):
                v={'development0':[1.,0.],'development1':[.9,np.sqrt(1-.9**2)],
                   'development2':[1.,0.],'development3':[.2,np.sqrt(1-.2**2)]}[p['id']]
                return {'shape':v}
        certificate=directory/'threshold.json';calibrate(manifest,certificate,Backend(),public_root=directory)
        return SelectionConfig(calibration_path=str(certificate),min_image_side=16)
    return factory


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
