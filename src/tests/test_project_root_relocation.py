"""项目复制后的访问路径合成验证；不重写冻结清单或调用实验。"""
import hashlib
import json
from pathlib import Path, PurePosixPath

import pytest

import src.paths as paths


OLD_WINDOWS = r'D:\Material\Literature Reading\code\perLage'


@pytest.fixture
def relocated_project(tmp_path, monkeypatch):
    project = tmp_path / 'copied_project'
    project.mkdir()
    monkeypatch.setattr(paths, 'PROJECT', project)
    monkeypatch.setenv('PERLAGE_ORIGINAL_PROJECT_ROOT', OLD_WINDOWS)
    return project


def test_explicit_windows_root_reads_actual_copy_without_changing_frozen_json(relocated_project):
    image = relocated_project / 'output/run/query.png'
    image.parent.mkdir(parents=True)
    image.write_bytes(b'copied image bytes')
    record = relocated_project / 'output/run/FROZEN.json'
    old_image = OLD_WINDOWS + r'\output\run\query.png'
    body = json.dumps({'image': old_image, 'sha256': hashlib.sha256(image.read_bytes()).hexdigest()},
                      ensure_ascii=False).encode('utf-8')
    record.write_bytes(body)
    before = hashlib.sha256(body).hexdigest()
    loaded = json.loads(paths.ProjectPath(OLD_WINDOWS + r'\output\run\FROZEN.json').read_text(encoding='utf-8'))
    assert loaded['image'] == old_image
    assert paths.ProjectPath(loaded['image']).read_bytes() == image.read_bytes()
    assert paths.relocate_io_path(loaded['image']) == str(image)
    assert hashlib.sha256(record.read_bytes()).hexdigest() == before


@pytest.mark.parametrize('old,new', [
    ('src/output/run/result.json', 'output/run/result.json'),
    ('src/data/private/query.json', 'output/datasets/private/query.json'),
    ('src/configs/evaluator.local.json', 'src/visual/config/evaluator.local.json'),
    ('src/configs/pipeline.local.json', 'src/experiments/config/pipeline.local.json'),
    ('src/data/indication_prompt_template_neutral.json', 'src/agent/prompts/templates/neutral.json'),
])
def test_original_root_mapping_composes_with_existing_relocations(relocated_project, old, new):
    assert paths.ProjectPath(OLD_WINDOWS + '/' + old) == relocated_project / new


def test_windows_root_uses_segment_boundaries_case_and_separator_semantics(relocated_project):
    old = r'd:/material/literature reading/CODE/PERLAGE/output/run.txt'
    assert paths.ProjectPath(old) == relocated_project / 'output/run.txt'
    assert paths.ProjectPath(OLD_WINDOWS, 'output', 'run.txt') == relocated_project / 'output/run.txt'
    for other in (
        r'D:\Material\Literature Reading\code\lage\output\run.txt',
        r'D:\Material\Literature Reading\code\MINJA\output\run.txt',
        OLD_WINDOWS + r'_backup\output\run.txt',
        r'D:\elsewhere\perLage\output\run.txt',
    ):
        assert paths.relocate_project_path(other) == other


@pytest.mark.parametrize('tail', [r'..\lage\secret.json', r'run\..\..\outside.json', '../outside.json'])
def test_original_root_escape_is_rejected(relocated_project, tail):
    with pytest.raises(ValueError, match='超出'):
        paths.ProjectPath(OLD_WINDOWS + '\\' + tail)


def test_safe_parent_step_stays_in_copied_root(relocated_project):
    assert paths.ProjectPath(OLD_WINDOWS + r'\output\discard\..\run.txt') == relocated_project / 'output/run.txt'


def test_mapping_enabled_preserves_native_absolute_and_relative_paths(relocated_project, monkeypatch):
    current = relocated_project / 'output/current.txt'
    assert paths.ProjectPath(current) == current
    assert paths.ProjectPath(relocated_project, 'src/output/current.txt') == current
    monkeypatch.chdir(relocated_project)
    assert paths.ProjectPath('output', 'current.txt') == Path('output/current.txt')
    assert paths.ProjectPath('src/output/current.txt') == current


def test_old_posix_root_maps_independently_of_host_platform(relocated_project, monkeypatch):
    monkeypatch.setenv('PERLAGE_ORIGINAL_PROJECT_ROOT', '/old/research/perlage')
    assert paths.relocate_project_path('/old/research/perlage/output/run.txt') == str(relocated_project / 'output/run.txt')
    assert paths.ProjectPath('/old/research/perlage', 'src/output/run.txt') == relocated_project / 'output/run.txt'
    assert paths.relocate_project_path('/old/research/perlage-copy/output/run.txt') == '/old/research/perlage-copy/output/run.txt'
    assert paths.relocate_project_path('/old/research/lage/output/run.txt') == '/old/research/lage/output/run.txt'


def test_windows_frozen_path_maps_before_linux_style_path_interpretation(monkeypatch):
    monkeypatch.setenv('PERLAGE_ORIGINAL_PROJECT_ROOT', OLD_WINDOWS)
    monkeypatch.setattr(paths, 'PROJECT', PurePosixPath('/srv/perlage'))
    monkeypatch.setattr(paths, '_Path', PurePosixPath)
    assert paths.relocate_project_path(OLD_WINDOWS + r'\output\run\query.png') == '/srv/perlage/output/run/query.png'
    assert paths.relocate_project_path(OLD_WINDOWS + r'\src\output\run\result.json') == '/srv/perlage/output/run/result.json'


def test_environment_unset_preserves_legacy_behavior_and_external_paths(relocated_project, monkeypatch):
    monkeypatch.delenv('PERLAGE_ORIGINAL_PROJECT_ROOT')
    original = OLD_WINDOWS + r'\src\output\frozen.json'
    assert paths.relocate_project_path(original) == original
    assert paths.ProjectPath(relocated_project / 'src/output/current.json') == relocated_project / 'output/current.json'
    sibling = relocated_project.parent / 'lage/src/output/frozen.json'
    assert paths.relocate_project_path(sibling) == str(sibling)
    assert paths.relocate_io_path(42) == 42


@pytest.mark.parametrize('root', ['relative/project', '/', 'D:\\', '/old/../perlage'])
def test_invalid_explicit_root_fails_instead_of_guessing(monkeypatch, root):
    monkeypatch.setenv('PERLAGE_ORIGINAL_PROJECT_ROOT', root)
    with pytest.raises(ValueError, match='绝对旧项目目录'):
        paths.relocate_project_path('output/run.json')
