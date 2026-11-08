"""从明确文件加载购物附加指导；该内容不是基础系统角色句。"""
from src.paths import ProjectPath as Path

from src.paths import PROMPTS
from src.visual.audit import read_json
from src.visual.image_utils import sha256_file


DEFAULT_TEMPLATE = PROMPTS / 'neutral.json'
DEFAULT_SETTINGS = PROMPTS.parent / 'settings.json'


def load_shopping_prompt(path=None, *, profile=None, expected_sha256=None, settings_path=None):
    """正文逐字加载；实验可指定配置名或独立文件，实际输入绑定文件摘要。"""
    settings = None
    if path is not None and profile is not None:
        raise ValueError('提示词文件和配置名只能指定其中一个')
    if path is None:
        settings = read_json(settings_path or DEFAULT_SETTINGS)
        profile = profile or settings['default_profile']
        if profile not in settings['profiles']:
            raise ValueError('未知购物提示词配置：' + profile)
        path = Path(settings_path or DEFAULT_SETTINGS).parent / settings['profiles'][profile]
    path = Path(path).resolve()
    actual_sha = sha256_file(path)
    if expected_sha256 is not None and actual_sha != expected_sha256:
        raise ValueError('购物附加指导文件与本轮冻结摘要不一致')
    values = read_json(path)
    if (not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], dict)
            or set(values[0]) != {'note1'} or not isinstance(values[0]['note1'], str)
            or not values[0]['note1'].strip()):
        raise ValueError('购物附加指导须为包含唯一非空 note1 的单元素数组')
    result = dict(text=values[0]['note1'], path=str(path), sha256=actual_sha,
                  role='auxiliary_user_guidance', base_system_prompt_unchanged=True)
    if settings is not None:
        result.update(profile=profile, settings_path=str(Path(settings_path or DEFAULT_SETTINGS).resolve()),
                      settings_sha256=sha256_file(settings_path or DEFAULT_SETTINGS))
    return result
