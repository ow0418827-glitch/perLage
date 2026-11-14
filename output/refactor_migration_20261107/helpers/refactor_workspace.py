"""One-time, workspace-local relocation. Historical artifact bytes stay untouched."""
from pathlib import Path
import json
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def move(old, new):
    source, destination = (ROOT / old).resolve(), (ROOT / new).resolve()
    if not source.is_relative_to(ROOT) or not destination.is_relative_to(ROOT):
        raise ValueError('Relocation escaped workspace')
    if source.exists():
        if destination.exists():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.rename(destination)


tracked = subprocess.check_output(['git', 'ls-files'], cwd=ROOT, text=True).splitlines()
snapshot = ROOT / 'tmp/pre_refactor'
for name in tracked:
    source = ROOT / name
    if source.is_file():
        destination = snapshot / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

for old, new in [
    ('src/output', 'output'),
    ('src/data/public', 'output/datasets/public'),
    ('src/data/private', 'output/datasets/private'),
    ('src/data/indication_prompt_template_neutral.json', 'src/agent/prompts/templates/neutral.json'),
    ('src/memory', 'src/agent/memory'),
    ('src/configuration', 'src/agent/config'),
    ('src/vision', 'src/visual'),
    ('src/data_sources', 'src/visual/selection'),
    ('src/environment', 'src/webshop'),
    ('src/retrieval', 'src/deprecated/retrieval'),
    ('src/defense/safe_recall', 'src/deprecated/safe_recall'),
    ('src/dependencies', 'requirements'),
]:
    move(old, new)

for path in (ROOT / 'src/configs').iterdir():
    if path.is_file():
        folder = ('src/agent/config' if path.name.startswith(('llm.', 'base_config'))
                  else 'src/visual/config' if path.name.startswith(('optimization', 'evaluator', 'shape_'))
                  else 'src/experiments/config')
        move(path.relative_to(ROOT), f'{folder}/{path.name}')

# Remove only now-empty package directories; never recurse over artifacts.
for name in ['src/configs', 'src/data', 'src/defense']:
    directory = ROOT / name
    for cache in directory.glob('__pycache__'):
        if cache.resolve().is_relative_to(ROOT):
            shutil.rmtree(cache)
    for stub in directory.glob('__init__.py'):
        stub.unlink()
    directory.rmdir()

(ROOT / 'src/deprecated/__init__.py').write_text('', encoding='utf-8')
(ROOT / 'src/visual/encoders').mkdir(exist_ok=True)
(ROOT / 'src/visual/_lucid').mkdir(exist_ok=True)
move('src/visual/joint_encoder.py', 'src/visual/encoders/joint.py')
move('src/visual/joint_encoder_worker.py', 'src/visual/encoders/worker.py')
move('src/vendor/dual_rag/encoder.py', 'src/visual/encoders/clip.py')
for old, new in [
    ('optimization/attack_core/surrogates.py', 'surrogates.py'),
    ('optimization/attack_core/objective.py', 'objective.py'),
    ('optimization/pixel_attack/update.py', 'pixel.py'),
    ('optimization/common/profiling.py', 'profiling.py'),
    ('model_paths.py', 'model_paths.py'),
]:
    move('src/vendor/dual_rag/' + old, 'src/visual/_lucid/' + new)
move('src/vendor/SSA_LICENSE', 'src/visual/_lucid/SSA_LICENSE')
move('src/vendor/NOTICE.md', 'src/visual/_lucid/NOTICE.md')
move('src/vendor/dual_rag/optimization/common/io.py', 'src/visual/io.py')

# Fold OT primitives into the objective, avoiding another nested attack package.
ot = (ROOT / 'src/vendor/dual_rag/optimization/attack_core/ot.py').read_text(encoding='utf-8')
objective = ROOT / 'src/visual/_lucid/objective.py'
body = objective.read_text(encoding='utf-8')
body = body.replace('from dual_rag.optimization.attack_core.ot import local_ot_similarity, token_kmeans', '')
ot = ot.replace('from __future__ import annotations', '')
ot = ot.replace('from dual_rag.optimization.common.numerics import assert_finite',
                (ROOT / 'src/vendor/dual_rag/optimization/common/numerics.py').read_text(encoding='utf-8').replace('from __future__ import annotations', ''))
future, rest = body.split('\n', 1)
objective.write_text(future + '\n' + ot + '\n' + rest, encoding='utf-8')

replacements = {
    'src.vision.joint_encoder_worker': 'src.visual.encoders.worker',
    'src.vision.joint_encoder': 'src.visual.encoders.joint',
    'src.vision.dual_rag_bridge': 'src.visual._lucid.pipeline',
    'src.vision': 'src.visual',
    'src.memory': 'src.agent.memory',
    'src.configuration': 'src.agent.config',
    'src.environment': 'src.webshop',
    'src.data_sources': 'src.visual.selection',
    'src.retrieval': 'src.deprecated.retrieval',
    'src.defense.safe_recall': 'src.deprecated.safe_recall',
    'dual_rag.optimization.attack_core.objective': 'src.visual._lucid.objective',
    'dual_rag.optimization.attack_core.surrogates': 'src.visual._lucid.surrogates',
    'dual_rag.optimization.attack_core.engine': 'src.visual._lucid.pipeline',
    'dual_rag.optimization.common.config': 'src.visual._lucid.config',
    'dual_rag.optimization.common.io': 'src.visual.io',
    'dual_rag.optimization.common.profiling': 'src.visual._lucid.profiling',
    'dual_rag.original_attack': 'src.visual._lucid.config',
    'dual_rag.encoder': 'src.visual.encoders.clip',
    'src/vision/joint_encoder_worker.py': 'src/visual/encoders/worker.py',
    'src/vision/joint_encoder.py': 'src/visual/encoders/joint.py',
    'src/vision/dual_rag_bridge.py': 'src/visual/_lucid/pipeline.py',
    'src/vision/': 'src/visual/',
    'src/data_sources/': 'src/visual/selection/',
    'src/memory/': 'src/agent/memory/',
    'src/configuration/': 'src/agent/config/',
    'src/environment/': 'src/webshop/',
    'src/output/': 'output/',
    'src/data/public/': 'output/datasets/public/',
    'src/data/private/': 'output/datasets/private/',
    'src/data/indication_prompt_template_neutral.json': 'src/agent/prompts/templates/neutral.json',
    'src/dependencies/': 'requirements/',
}
files = [*list((ROOT / 'src').rglob('*.py')), *list((ROOT / 'tools').glob('*.py'))]
files += [ROOT / 'run_pipeline.ps1', ROOT / 'README.md']
files += list((ROOT / 'docs').glob('*.md'))
for path in files:
    if 'vendor' in path.parts or '__pycache__' in path.parts:
        continue
    text = path.read_text(encoding='utf-8')
    for old, new in replacements.items():
        text = text.replace(old, new)
    # Standard package imports everywhere; history-path translation is centralized.
    if path.name != 'paths.py' and path.parent.name != 'tools':
        text = text.replace('from pathlib import Path', 'from src.paths import ProjectPath as Path')
    path.write_text(text, encoding='utf-8')

for folder in ['src/agent/config', 'src/visual/config', 'src/experiments/config']:
    for path in (ROOT / folder).glob('*'):
        if path.suffix not in ['.json', '.yaml'] or '.local.' in path.name:
            continue
        text = path.read_text(encoding='utf-8')
        for old, new in replacements.items():
            text = text.replace(old, new)
        path.write_text(text, encoding='utf-8')

# Vendor code has been retained in the local ignored snapshot for regression only.
vendor = (ROOT / 'src/vendor').resolve()
if not vendor.is_relative_to(ROOT) or vendor == ROOT:
    raise ValueError('Invalid obsolete vendor target')
shutil.rmtree(vendor)
for folder in ['src/visual/encoders', 'src/visual/_lucid', 'data/groups']:
    (ROOT / folder).mkdir(parents=True, exist_ok=True)
    if folder.startswith('src/'):
        (ROOT / folder / '__init__.py').write_text('', encoding='utf-8')
(ROOT / 'data/groups/.gitkeep').write_text('', encoding='utf-8')
print('Workspace relocation complete; original artifact bytes preserved.')
