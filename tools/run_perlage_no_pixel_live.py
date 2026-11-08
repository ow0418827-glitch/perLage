import json
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

from src.paths import PROJECT, OUTPUT, ProjectPath as Path

from src.visual.optimization import generate_attack
from src.visual.catalog_training_guard import catalog_training_guard
from src.visual.audit import read_json, freeze_json
from src.visual.image_utils import sha256_file
from src.visual.attack_completion import validate_attack_completion
from src.visual._perlage_no_pixel.evidence import validate_complete, METHOD, DEFINITION

print(f"=== Starting live run for {METHOD} ===", flush=True)
destination = OUTPUT / 'perlage_no_pixel_live_credential'
destination.mkdir(parents=True, exist_ok=True)

data_dir = OUTPUT / 'shampoo_ablation_20261107/data/07_shampoo'
source_img = data_dir / 'source.jpg'
target_img = data_dir / 'target.jpg'
base_def_path = data_dir / 'latent_only_matched.json'

base_def = read_json(base_def_path)
definition = json.loads(json.dumps(base_def))

definition['method'] = METHOD
definition['definition'] = DEFINITION
definition['pixel_optimization'] = False
definition['data']['source'] = str(source_img.resolve())
definition['data']['target'] = str(target_img.resolve())

definition.setdefault('initialization', {}).setdefault('optimizer', {})
definition['initialization']['optimizer']['shadow_weight'] = 0.0
definition['initialization']['optimizer']['pixel_clean_steps'] = 0
definition['initialization']['optimizer']['pixel_diffusion_steps'] = 0
definition['initialization']['optimizer']['pixel_levels'] = 0
definition['initialization']['optimizer']['check_every'] = 20
definition['initialization']['optimizer']['latent_clean_steps'] = 20
definition['initialization']['optimizer']['latent_diffusion_steps'] = 20

definition.setdefault('latent', {})
definition['latent']['auxiliary_levels'] = 0
definition['latent']['steps'] = 20
definition['latent']['check_every'] = 10

start_time = time.perf_counter()
with catalog_training_guard(allowed_output_roots=[destination, data_dir]) as audit:
    image, metadata = generate_attack(
        source=source_img,
        target=target_img,
        output_dir=destination,
        method=METHOD,
        perlage_no_pixel_config=definition,
        device='cuda',
    )
    freeze_json(destination / 'completion.json', metadata)

audit_clean = {k: v for k, v in audit.items() if k != 'active'}
freeze_json(destination / 'TRAINING_ACCESS_AUDIT.json', audit_clean)
elapsed = time.perf_counter() - start_time
print(f"=== Pipeline generation complete in {elapsed:.2f} s ===", flush=True)

print("=== Validating 4-stage evidence and ready record ===", flush=True)
ready = read_json(destination / 'ready.json')
validate_complete(ready, image, method=METHOD)
print("validate_complete SUCCEEDED!", flush=True)

print("=== Validating attack completion ===", flush=True)
validate_attack_completion(
    metadata,
    requested_steps=definition['latent']['steps'],
    expected={
        'source_image_sha256': sha256_file(source_img),
        'target_image_sha256': sha256_file(target_img),
    }
)
print("validate_attack_completion SUCCEEDED!", flush=True)

summary = {
    'method': METHOD,
    'definition': DEFINITION,
    'output_image': str(image),
    'output_image_sha256': sha256_file(image),
    'generative_anchor_sha256': sha256_file(destination / 'generative_anchor.png'),
    'ready_sha256': sha256_file(destination / 'ready.json'),
    'method_identity_sha256': sha256_file(destination / 'method_identity.json'),
    'audit_sha256': sha256_file(destination / 'TRAINING_ACCESS_AUDIT.json'),
    'completion_sha256': sha256_file(destination / 'completion.json'),
    'elapsed_seconds': elapsed,
    'stages': list(ready['stages'].keys()),
    'pixel_optimization': ready['pixel_optimization'],
    'auxiliary_max_levels': ready['auxiliary_max_levels'],
}
freeze_json(destination / 'LIVE_RUN_SUMMARY.json', summary)
print(json.dumps(summary, indent=2, ensure_ascii=False))
