"""从明确文件加载购物附加指导；该内容不是基础系统角色句。"""
from pathlib import Path

from src.paths import DATA
from src.vision.audit import read_json
from src.vision.image_utils import sha256_file


DEFAULT_TEMPLATE = DATA / 'indication_prompt_template_neutral.json'


def load_shopping_prompt(path=None, *, expected_sha256=None):
    path = Path(path or DEFAULT_TEMPLATE).resolve()
    actual_sha = sha256_file(path)
    if expected_sha256 is not None and actual_sha != expected_sha256:
        raise ValueError('购物附加指导文件与本轮冻结摘要不一致')
    values = read_json(path)
    if (not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], dict)
            or set(values[0]) != {'note1'} or not isinstance(values[0]['note1'], str)
            or not values[0]['note1'].strip()):
        raise ValueError('购物附加指导须为包含唯一非空 note1 的单元素数组')
    return dict(text=values[0]['note1'], path=str(path), sha256=actual_sha,
                role='auxiliary_user_guidance', base_system_prompt_unchanged=True)
