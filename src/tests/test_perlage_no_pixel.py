"""针对独立无像素扰动感知优化对抗方法 (PerLAGE_no_pixel) 的专用自动化测试套件。

涵盖：
1. 方法独立注册与方法定义标识核验
2. 命令行参数解析与互斥/跨配置检查
3. 专用流水线与初始化零像素、零残差约束硬拒绝
4. 四阶段完整凭证与完成记录严格审计
5. validate_attack_completion 统一完成验证集成
"""
import argparse
import copy
from src.paths import ProjectPath as Path
import pytest

from src.visual.audit import save_json, read_json
from src.visual.image_utils import sha256_file
from src.visual.retrieval_objective import TRAINING_DATA_POLICY
from src.visual.optimization import (
    METHODS,
    NATURAL_METHODS,
    normalize_method,
    add_method_arguments,
    options_from_args,
    generate_attack,
)
from src.visual._perlage.components.evidence import STAGES
from src.visual._perlage_no_pixel import (
    METHOD as NO_PIXEL_METHOD,
    DEFINITION as NO_PIXEL_DEFINITION,
    validate_complete,
)
from src.visual._perlage_no_pixel.evidence import definition_for_method
from src.visual._perlage_no_pixel.pipeline import (
    PerLAGENoPixelConfig,
    validate_definition,
    initialization_config,
)
from src.visual.attack_completion import validate_attack_completion


# ==============================================================================
# 1. 方法注册、定义标识与方法规范化测试
# ==============================================================================

def test_methods_registration():
    """验证 PerLAGE_no_pixel 在 METHODS 与 NATURAL_METHODS 中正式注册，三者完全平级。"""
    assert 'PerLAGE_no_pixel' in METHODS
    assert METHODS == ('lucid', 'PerLAGE', 'PerLAGE_no_pixel')
    assert 'PerLAGE_no_pixel' in NATURAL_METHODS
    assert NATURAL_METHODS == ('PerLAGE', 'PerLAGE_no_pixel')


def test_normalize_method_support():
    """验证大小写与变体能准确规范化为 PerLAGE_no_pixel，非法方法拒绝。"""
    assert normalize_method('PerLAGE_no_pixel') == 'PerLAGE_no_pixel'
    assert normalize_method('perlage_no_pixel') == 'PerLAGE_no_pixel'
    with pytest.raises(ValueError, match='方法必须是 lucid、PerLAGE 或 PerLAGE_no_pixel'):
        normalize_method('unknown_attack_method')


def test_definition_registration():
    """验证 PerLAGE_no_pixel 独立包中的常量及 definition_for_method 映射。"""
    assert NO_PIXEL_METHOD == 'PerLAGE_no_pixel'
    assert NO_PIXEL_DEFINITION == 'perlage_no_pixel_foreground_statistics_harmonization_perceptual_latent_v1'
    assert definition_for_method('PerLAGE_no_pixel') == NO_PIXEL_DEFINITION
    assert definition_for_method('perlage_no_pixel') == NO_PIXEL_DEFINITION
    assert definition_for_method() == NO_PIXEL_DEFINITION

    with pytest.raises(ValueError, match='未知的无像素感知方法身份'):
        definition_for_method('PerLAGE')
    with pytest.raises(ValueError, match='未知的无像素感知方法身份'):
        definition_for_method('lucid')


# ==============================================================================
# 2. 命令行参数解析与跨方法参数互斥测试
# ==============================================================================

def test_cli_argument_parsing_success():
    """验证 --method PerLAGE_no_pixel 与 --perlage-no-pixel-config 参数正常解析。"""
    parser = argparse.ArgumentParser()
    add_method_arguments(parser)
    args = parser.parse_args(['--method', 'PerLAGE_no_pixel', '--perlage-no-pixel-config', 'config.json'])
    assert args.method == 'PerLAGE_no_pixel'
    assert args.perlage_no_pixel_config == 'config.json'
    assert args.perlage_config is None

    options = options_from_args(args)
    assert options == {'perlage_no_pixel_config': 'config.json'}


def test_cli_group_parameter_removed():
    """验证已移除按组分派配置与记忆的机制，传入 group 必须报错。"""
    parser = argparse.ArgumentParser()
    add_method_arguments(parser)
    args = parser.parse_args(['--method', 'PerLAGE_no_pixel', '--perlage-no-pixel-config', 'config.json'])
    with pytest.raises(ValueError, match='已移除按组分派配置和记忆的机制'):
        options_from_args(args, group='07_shampoo')


def test_cli_cross_configuration_rejected():
    """验证无像素配置与有像素感知方法配置之间的互斥约束。"""
    parser = argparse.ArgumentParser()
    add_method_arguments(parser)

    # PerLAGE 读取了无像素配置
    args_wrong_1 = parser.parse_args(['--method', 'PerLAGE', '--perlage-no-pixel-config', 'no_pixel.json'])
    with pytest.raises(ValueError, match='无像素方法配置只能用于 PerLAGE_no_pixel'):
        options_from_args(args_wrong_1)

    # PerLAGE_no_pixel 读取了常规感知配置
    args_wrong_2 = parser.parse_args(['--method', 'PerLAGE_no_pixel', '--perlage-config', 'standard.json'])
    with pytest.raises(ValueError, match='生成配置只能用于对应的感知方法'):
        options_from_args(args_wrong_2)


def test_generate_attack_configuration_exclusivity(tmp_path):
    """验证 generate_attack 统一接口对无像素方法的互斥与边界保护。"""
    src = tmp_path / 'source.png'
    tgt = tmp_path / 'target.png'
    src.write_bytes(b'fake_src')
    tgt.write_bytes(b'fake_tgt')
    out = tmp_path / 'output'

    # PerLAGE 不能传入 perlage_no_pixel_config
    with pytest.raises(ValueError, match='完整感知方法不能读取无像素方法配置'):
        generate_attack(src, tgt, out, method='PerLAGE', perlage_no_pixel_config={'a': 1})

    # PerLAGE_no_pixel 不能传入 perlage_config
    with pytest.raises(ValueError, match='无像素方法必须使用其独立配置参数'):
        generate_attack(src, tgt, out, method='PerLAGE_no_pixel', perlage_config={'a': 1})

    # PerLAGE_no_pixel 缺少独立配置
    with pytest.raises(ValueError, match='感知方法缺少本次筛选图对的公开配置'):
        generate_attack(src, tgt, out, method='PerLAGE_no_pixel')

    # PerLAGE_no_pixel 禁止传入 target_text
    with pytest.raises(ValueError, match='感知方法只从其公开配置读取商品名和通用训练指令'):
        generate_attack(src, tgt, out, method='PerLAGE_no_pixel', perlage_no_pixel_config={'a': 1}, target_text='attack')

    # lucid 不能传入 perlage_no_pixel_config
    with pytest.raises(ValueError, match='像素方法不能读取感知方法配置'):
        generate_attack(src, tgt, out, method='lucid', perlage_no_pixel_config={'a': 1})


# ==============================================================================
# 3. 流水线定义与初始化配置零像素/零残差约束测试
# ==============================================================================

def make_valid_definition(products):
    """构建符合 PerLAGE_no_pixel 规范的基准 definition 字典。"""
    return {
        'method': NO_PIXEL_METHOD,
        'definition': NO_PIXEL_DEFINITION,
        'color_scheme': 'statistics',
        'training_data_policy': TRAINING_DATA_POLICY,
        'data': {
            'source': products[1]['image'],
            'target': products[0]['image'],
            'source_title': products[1]['title'],
            'training_queries': ['q1', 'q2', 'q3'],
        },
        'source_boxes': [[0, 0, 10, 10]],
        'target_boxes': [[0, 0, 10, 10]],
        'print_regions': [[0.1, 0.1, 0.2, 0.2]],
        'initialization': {
            'optimizer': {
                'pixel_clean_steps': 0,
                'pixel_diffusion_steps': 0,
                'pixel_levels': 0,
                'shadow_weight': 0.0,
            },
            'generation': {'steps': 30},
        },
        'harmonization': {
            'smoothing_sigma': 12.0,
            'print_ring_sigma': 7.0,
            'selected_color_mix': 0.25,
            'strength': 0.75,
            'blend_mode': 'matte',
        },
        'latent': {
            'steps': 40,
            'check_every': 10,
            'auxiliary_levels': 0,
        },
        'pixel_optimization': False,
    }


def test_validate_definition_valid(products):
    """验证符合规范的 PerLAGE_no_pixel 定义能够成功通过 validate_definition。"""
    definition = make_valid_definition(products)
    config = validate_definition(definition)
    assert isinstance(config, PerLAGENoPixelConfig)
    assert config.pixel_optimization is False
    assert config.latent['auxiliary_levels'] == 0


def test_validate_definition_pixel_optimization_rejected(products):
    """验证开启 pixel_optimization 时被立即拦截拒绝。"""
    definition = make_valid_definition(products)
    definition['pixel_optimization'] = True
    with pytest.raises(ValueError, match='无像素方法禁止开启像素优化'):
        validate_definition(definition)


def test_validate_definition_auxiliary_levels_rejected(products):
    """验证潜变量配置中包含 auxiliary_levels > 0 时被立即拦截拒绝。"""
    definition = make_valid_definition(products)
    definition['latent']['auxiliary_levels'] = 4
    with pytest.raises(ValueError, match='仅潜变量模式禁止像素优化：固定像素残差'):
        validate_definition(definition)


@pytest.mark.parametrize('param,val', [
    ('pixel_clean_steps', 20),
    ('pixel_diffusion_steps', 60),
    ('pixel_levels', 4),
])
def test_validate_definition_optimizer_pixel_steps_rejected(products, param, val):
    """验证初始化优化器中配置非零像素步数或等级时被拦截拒绝。"""
    definition = make_valid_definition(products)
    definition['initialization']['optimizer'][param] = val
    with pytest.raises(ValueError, match='无像素模式不能配置像素优化'):
        validate_definition(definition)


def test_validate_definition_forbidden_keys(products):
    """验证禁止竞争池、旧成图或分支选择键名。"""
    definition = make_valid_definition(products)
    definition['public_pool'] = 'pool.json'
    with pytest.raises(ValueError, match='优化方法禁止竞争池、旧成图与分支选择'):
        validate_definition(definition)


def test_initialization_config_zero_pixel_enforcement():
    """验证 initialization_config 严格检查零像素优化步数。"""
    valid_def = {
        'method': NO_PIXEL_METHOD,
        'initialization': {
            'optimizer': {
                'pixel_clean_steps': 0,
                'pixel_diffusion_steps': 0,
                'pixel_levels': 0,
            }
        }
    }
    cfg = initialization_config(valid_def)
    assert cfg.pixel_clean_steps == 0
    assert cfg.pixel_diffusion_steps == 0
    assert cfg.pixel_levels == 0

    invalid_def = copy.deepcopy(valid_def)
    invalid_def['initialization']['optimizer']['pixel_clean_steps'] = 10
    with pytest.raises(ValueError, match='仅潜变量模式不能配置像素优化：pixel_clean_steps'):
        initialization_config(invalid_def)


# ==============================================================================
# 4. 证据模块 (evidence.py) 校验逻辑测试
# ==============================================================================

def make_valid_ready_fixture(products, tmp_path):
    """构建一份完整的四阶段合格 ready 证据字典。"""
    proof = tmp_path / 'run_identity.json'
    save_json(proof, {'public_binding': {'shadow_pool_used': False}})

    opt_identity = tmp_path / 'optimizer_identity.json'
    save_json(opt_identity, {
        'config': {
            'pixel_clean_steps': 0,
            'pixel_diffusion_steps': 0,
            'pixel_levels': 0,
        }
    })

    ready_init = tmp_path / 'initialization_ready.json'
    save_json(ready_init, {'status': 'completed'})

    stages = {}
    for stage_name in STAGES:
        file_path = tmp_path / f'{stage_name}_artifact.dat'
        file_path.write_bytes(f'stage_{stage_name}'.encode('utf-8'))
        stages[stage_name] = {
            'status': 'completed',
            'files': {str(file_path): sha256_file(file_path)},
        }
    stages['latent_generation_refinement'].update(steps_completed=40, steps_requested=40)

    final_img = tmp_path / 'final_adversarial.png'
    final_img.write_bytes(Path(products[1]['image']).read_bytes())

    method_id = tmp_path / 'method_identity.json'
    config_digest = 'a' * 64
    save_json(method_id, {
        'method': NO_PIXEL_METHOD,
        'definition': NO_PIXEL_DEFINITION,
        'configuration_sha256': config_digest,
        'source_sha256': products[1]['image_sha256'],
        'target_sha256': products[0]['image_sha256'],
    })

    return {
        'method': NO_PIXEL_METHOD,
        'definition': NO_PIXEL_DEFINITION,
        'status': 'completed',
        'color_scheme': 'statistics',
        'shadow_pool_used': False,
        'purchase_history_used': False,
        'victim_memory_used': False,
        'formal_queries_used': False,
        'training_data_policy': TRAINING_DATA_POLICY,
        'configuration_sha256': config_digest,
        'method_identity': str(method_id),
        'method_identity_sha256': sha256_file(method_id),
        'source_sha256': products[1]['image_sha256'],
        'target_sha256': products[0]['image_sha256'],
        'pixel_optimization': False,
        'auxiliary_max_levels': 0,
        'latent_config': {'auxiliary_levels': 0},
        'optimization_mode': 'latent_only',
        'stages': stages,
        'initialization_identity': str(proof),
        'initialization_optimizer_identity': str(opt_identity),
        'initialization_evidence': {
            str(proof): sha256_file(proof),
            str(opt_identity): sha256_file(opt_identity),
            str(ready_init): sha256_file(ready_init),
        },
        'candidates': {
            'final': {
                'image': str(final_img),
                'sha256': sha256_file(final_img),
            }
        }
    }, final_img


def test_evidence_validate_complete_success(products, tmp_path):
    """验证标准的 PerLAGE_no_pixel 完成凭证可以通过 validate_complete。"""
    ready, final_img = make_valid_ready_fixture(products, tmp_path)
    res = validate_complete(ready, final_img)
    assert res['method'] == NO_PIXEL_METHOD
    assert res['definition'] == NO_PIXEL_DEFINITION
    assert res['pixel_optimization'] is False
    assert res['auxiliary_max_levels'] == 0


def test_evidence_rejects_pixel_optimization(products, tmp_path):
    """验证 ready 记录中 pixel_optimization 不为 False 时被拒绝。"""
    ready, final_img = make_valid_ready_fixture(products, tmp_path)
    ready['pixel_optimization'] = True
    with pytest.raises(ValueError, match='无像素方法不能包含像素优化'):
        validate_complete(ready, final_img)

    ready['pixel_optimization'] = None
    with pytest.raises(ValueError, match='无像素方法不能包含像素优化'):
        validate_complete(ready, final_img)


def test_evidence_rejects_auxiliary_levels(products, tmp_path):
    """验证 ready 记录中存在非零 auxiliary_max_levels 时被拒绝。"""
    ready, final_img = make_valid_ready_fixture(products, tmp_path)
    ready['auxiliary_max_levels'] = 2
    with pytest.raises(ValueError, match='无像素方法不能包含固定像素残差'):
        validate_complete(ready, final_img)

    ready['auxiliary_max_levels'] = 0
    ready['latent_config']['auxiliary_levels'] = 2
    with pytest.raises(ValueError, match='无像素方法潜变量配置不能包含辅助像素预算'):
        validate_complete(ready, final_img)


def test_evidence_rejects_invalid_optimization_mode(products, tmp_path):
    """验证 optimization_mode 必须为 latent_only 或 None。"""
    ready, final_img = make_valid_ready_fixture(products, tmp_path)
    ready['optimization_mode'] = 'pixel_and_latent'
    with pytest.raises(ValueError, match='无像素方法优化模式必须为 latent_only'):
        validate_complete(ready, final_img)


def test_evidence_rejects_optimizer_non_zero_pixel(products, tmp_path):
    """验证 initialization_optimizer_identity 包含非零像素步数时被拒绝。"""
    ready, final_img = make_valid_ready_fixture(products, tmp_path)
    opt_path = Path(ready['initialization_optimizer_identity'])
    save_json(opt_path, {'config': {'pixel_clean_steps': 20, 'pixel_diffusion_steps': 0, 'pixel_levels': 0}})
    # 更新 sha256 避免提前在 sha256 检查中退出
    ready['initialization_evidence'][str(opt_path)] = sha256_file(opt_path)

    with pytest.raises(ValueError, match='无像素方法不能包含像素优化步数或等级'):
        validate_complete(ready, final_img)


def test_evidence_rejects_mismatched_method_or_definition(products, tmp_path):
    """验证方法身份与方法定义标识必须严格匹配。"""
    ready, final_img = make_valid_ready_fixture(products, tmp_path)
    ready['method'] = 'PerLAGE'
    with pytest.raises(ValueError, match='方法身份不符合无像素感知方法'):
        validate_complete(ready, final_img)

    ready['method'] = NO_PIXEL_METHOD
    ready['definition'] = 'perlage_foreground_statistics_harmonization_perceptual_latent_v1'
    with pytest.raises(ValueError, match='完成记录中的方法定义不符合'):
        validate_complete(ready, final_img)


@pytest.mark.parametrize('stage', STAGES)
def test_evidence_missing_stage_rejected(products, tmp_path, stage):
    """验证缺少四个自然优先阶段中任何一个阶段都不能标记为完成。"""
    ready, final_img = make_valid_ready_fixture(products, tmp_path)
    del ready['stages'][stage]
    with pytest.raises(ValueError, match='优化方法仍须完成四个自然优先阶段'):
        validate_complete(ready, final_img)


def test_evidence_incomplete_latent_steps_rejected(products, tmp_path):
    """验证潜变量细化步数未跑满时被拒绝。"""
    ready, final_img = make_valid_ready_fixture(products, tmp_path)
    ready['stages']['latent_generation_refinement']['steps_completed'] = 39
    ready['stages']['latent_generation_refinement']['steps_requested'] = 40
    with pytest.raises(ValueError, match='优化方法生成潜变量细化日程未完成'):
        validate_complete(ready, final_img)


def test_evidence_tampered_stage_file_rejected(products, tmp_path):
    """验证阶段文件哈希被篡改时被拒绝。"""
    ready, final_img = make_valid_ready_fixture(products, tmp_path)
    first_file = next(iter(ready['stages']['foreground_segmentation']['files'].keys()))
    Path(first_file).write_bytes(b'tampered_content')
    with pytest.raises(ValueError, match='优化方法阶段证据缺失或改变'):
        validate_complete(ready, final_img)


# ==============================================================================
# 5. validate_attack_completion 集成测试
# ==============================================================================

def test_validate_attack_completion_integration(products, tmp_path):
    """验证 validate_attack_completion 能正确分派并校验 PerLAGE_no_pixel。"""
    ready, final_img = make_valid_ready_fixture(products, tmp_path)
    metadata = {
        'method': NO_PIXEL_METHOD,
        'status': 'success',
        'output_image': str(final_img),
        'output_image_sha256': sha256_file(final_img),
        'source_image_sha256': products[1]['image_sha256'],
        'target_image_sha256': products[0]['image_sha256'],
        'perlage_ready': ready,
        'shadow_pool_used': False,
    }

    # 正常校验通过
    validate_attack_completion(metadata, requested_steps=40)

    # 错误1：shadow_pool_used 为 True
    bad_shadow = dict(metadata, shadow_pool_used=True)
    with pytest.raises(ValueError, match='完整方法不得读入影子池或私有共享记忆池'):
        validate_attack_completion(bad_shadow, requested_steps=40)

    # 错误2：图对摘要不匹配
    bad_pair = dict(metadata, source_image_sha256='wrong_hash')
    with pytest.raises(ValueError, match='四阶段证据与本次公开图对不符'):
        validate_attack_completion(bad_pair, requested_steps=40)

    # 错误3：状态不是 success
    bad_status = dict(metadata, status='failed')
    with pytest.raises(ValueError, match='图片方法尚未正常完成'):
        validate_attack_completion(bad_status, requested_steps=40)

    # 错误4：最终图片哈希不匹配
    bad_img = dict(metadata, output_image_sha256='wrong_hash')
    with pytest.raises(ValueError, match='最终图片缺失或摘要改变'):
        validate_attack_completion(bad_img, requested_steps=40)
