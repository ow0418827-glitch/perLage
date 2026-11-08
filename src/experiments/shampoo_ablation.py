"""洗发水冻结图对的像素预算与潜变量消融；生成进程不读取共享库。"""
import argparse
import os
from src.paths import ProjectPath as Path, PROJECT, OUTPUT
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

from src.visual.audit import read_json, freeze_json, save_json
from src.visual.image_utils import sha256_file

DEFAULT_ROOT = OUTPUT / 'shampoo_ablation_20261107'


def exact_copy(source, destination):
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copyfile(source, destination)
    if sha256_file(source) != sha256_file(destination):
        raise ValueError('输入副本与冻结文件摘要不一致')
    return destination


def prepare(root):
    root = Path(root)
    legacy = PROJECT.parent / 'lage/src/output/lage_runs'
    data = root / 'data/07_shampoo'
    public = OUTPUT / 'color_full_pipeline_20261106/public_inputs/07_shampoo'
    definition = read_json(public / 'definition.json')
    for role in ('source', 'target'):
        definition['data'][role] = str(exact_copy(public / (role + '.jpg'), data / (role + '.jpg')))
    freeze_json(data / 'definition.json', definition)
    matched = read_json(legacy / 'catalog_shampoo_no_shadow_20261029/definitions/07_shampoo.json')
    matched['method'] = definition['method']
    matched['definition'] = definition['definition']
    matched.pop('group', None)
    for role in ('source', 'target'):
        matched['data'][role] = definition['data'][role]
    matched['pixel_optimization'] = False
    matched['initialization']['optimizer'].update(pixel_clean_steps=0, pixel_diffusion_steps=0, pixel_levels=0)
    original_init = read_json(legacy / 'catalog_shampoo_no_shadow_20261029/generated/07_shampoo/initialization/run_identity.json')
    matched['initialization']['plan'] = original_init['plan']
    matched['initialization']['generation'] = original_init['generation']
    matched['latent']['auxiliary_levels'] = 0
    freeze_json(data / 'latent_only_matched.json', matched)
    freeze_json(data / 'CONTROL_PROTOCOL.json', dict(primary_baseline='perlage_pixel_control',
        historical_baseline='perlage_original_no_shadow', shared_current_latent_prefix=True,
        same_public_pair=True, same_public_title=True, same_training_queries=True,
        same_generation_plan_and_seed=True, same_latent_schedule_and_quality_limits=True,
        changed_factors=['remove_initialization_pixel_stages', 'remove_fixed_auxiliary_pixel_residual'],
        final_latent_steps=matched['latent']['steps'], initialization_latent_steps=80,
        later_color_revision_is_secondary_reference=True, shopping_runs=0))
    assets = {}
    lucid_record = read_json(legacy / 'lucid_shampoo_full_retest_20261027/groups/07_shampoo/attack_ready.json')
    if lucid_record['settings']['steps'] != 1000 or lucid_record['settings']['epsilon_levels'] != 16:
        raise ValueError('旧原始像素方法不是完整千步十六级对照')
    for role in ('source', 'target'):
        if lucid_record['settings'][role + '_sha256'] != sha256_file(data / (role + '.jpg')):
            raise ValueError('旧像素方法的图对不同')
    assets['lucid_original_16'] = exact_copy(lucid_record['image'], root / 'images/lucid_original_16.png')
    freeze_json(root / 'provenance/lucid_original.json', lucid_record)
    for name, folder in [('perlage_original_no_shadow', 'catalog_shampoo_no_shadow_20261029'),
                         ('perlage_legacy_reference', 'catalog_shampoo_20261028')]:
        ready_path = legacy / folder / 'generated/07_shampoo/ready.json'
        ready = read_json(ready_path)
        final = ready['candidates']['final']
        if ready['status'] != 'completed' or sha256_file(final['image']) != final['sha256']:
            raise ValueError('历史完整感知图片不符')
        if any(ready.get(role+'_sha256')!=sha256_file(data/(role+'.jpg')) for role in ('source','target')):
            raise ValueError('历史感知图没有使用同一冻结图对')
        for stage in ready['stages'].values():
            if stage['status']!='completed' or any(sha256_file(p)!=s for p,s in stage['files'].items()):
                raise ValueError('历史感知图的完整阶段证据不符')
        assets[name] = exact_copy(final['image'], root / 'images' / (name + '.png'))
        exact_copy(ready_path, root / 'provenance' / (name + '.json'))
    # 后来主色方案具有额外的颜色与边缘门槛变化，单列参考，不作为像素消融控制。
    newer = OUTPUT / 'color_full_pipeline_20261106/runs/dominant/07_shampoo/generation/ready.json'
    if newer.is_file():
        ready = read_json(newer)
        assets['perlage_color_revision_reference'] = exact_copy(
            ready['candidates']['final']['image'], root / 'images/perlage_color_revision_reference.png')
        exact_copy(newer, root / 'provenance/perlage_color_revision_reference.json')
    # 评估输入独立冻结，不交给生成进程。
    exact_copy(OUTPUT / 'shampoo_k13_20261106/inputs/07_shampoo/query.png', root / 'evaluation_inputs/query.png')
    freeze_json(root / 'HISTORICAL_IMAGES.json', {k: dict(image=str(v), sha256=sha256_file(v)) for k, v in assets.items()})
    print('公开图对和历史完整图已冻结。', flush=True)


def generate_lucid(root, levels):
    import torch
    from src.visual.catalog_training_guard import catalog_training_guard
    from src.visual.optimization import generate_attack
    root = Path(root)
    torch.set_num_threads(4)
    destination = root / f'generation/lucid_{levels}'
    with catalog_training_guard(allowed_output_roots=[root / 'data', destination]) as audit:
        image, metadata = generate_attack(root / 'data/07_shampoo/source.jpg', root / 'data/07_shampoo/target.jpg',
            destination, method='lucid', steps=1000, epsilon_levels=levels, device='cuda')
        freeze_json(destination / 'completion.json', metadata)
    freeze_json(destination / 'TRAINING_ACCESS_AUDIT.json', {k:v for k,v in audit.items() if k != 'active'})
    return image


def generate_perlage(root):
    import torch
    from src.visual.catalog_training_guard import catalog_training_guard
    from src.visual.optimization import generate_attack
    root = Path(root)
    torch.set_num_threads(4)
    destination = root / 'generation/perlage_latent_only'
    with catalog_training_guard(allowed_output_roots=[root / 'data', destination]) as audit:
        definition = read_json(root / 'data/07_shampoo/latent_only_matched.json')
        image,metadata = generate_attack(definition['data']['source'],definition['data']['target'],destination,
            method='PerLAGE',perlage_config=definition,device='cuda')
        ready = metadata['perlage_ready']
        optimizer = read_json(destination / 'initialization/attack/metadata.json')
        if ready['optimization_mode'] != 'latent_only' or ready['auxiliary_max_levels'] != 0:
            raise ValueError('仅潜变量消融出现了像素残差')
        if optimizer['pixel_residual_max_levels'] != 0 or any(
                row['phase'].startswith('pixel') for row in optimizer['history']):
            raise ValueError('仅潜变量消融执行了像素优化')
        freeze_json(destination / 'completion.json',metadata)
    freeze_json(destination / 'TRAINING_ACCESS_AUDIT.json',{k:v for k,v in audit.items() if k != 'active'})
    return image


def generate_pixel_control(root):
    """从本轮已完成的公共潜变量前缀分叉，补跑像素分支及完整最终细化。"""
    import torch
    from dataclasses import asdict
    from src.visual.catalog_training_guard import catalog_training_guard
    from src.visual.optimization import generate_attack
    from src.visual._perlage.pipeline import initialization_config
    root=Path(root)
    torch.set_num_threads(4)
    source=root/'generation/perlage_latent_only'
    parent=source/'initialization'
    destination=root/'generation/perlage_pixel_control'
    prefix=read_json(parent/'ready.json')
    prefix_meta=read_json(parent/'attack/metadata.json')
    if prefix_meta['status']!='completed' or prefix_meta['pixel_residual_max_levels']!=0:
        raise ValueError('配对对照需要已完整完成且没有像素扰动的公共潜变量前缀')
    amendment=root/'data/07_shampoo/PAIRED_PROTOCOL_AMENDMENT.json'
    if not amendment.exists():
        if (root/'evaluation/PLAN.json').exists():
            raise ValueError('正式评估开始后不得新增配对协议')
        freeze_json(amendment,dict(schema_version=1,frozen_at=datetime.now(timezone.utc).isoformat(),
            reason='Use byte-identical current latent prefix to isolate pixel stages from regeneration drift.',
            primary_perlage_control='perlage_pixel_control',historical_comparison_retained=True,
            common_prefix_reused_not_regenerated=True,reused_instruction_steps=30,reused_latent_steps=80,
            new_pixel_steps=[20,60],new_final_latent_steps=40,
            only_configuration_changes=['pixel_optimization','initialization.optimizer.pixel_clean_steps',
                'initialization.optimizer.pixel_diffusion_steps','initialization.optimizer.pixel_levels',
                'latent.auxiliary_levels'],total_variants=9,queries_per_variant=20,global_memory_count=189,
            full_scores_to_recompute=34020,formal_evaluation_started=False,
            common_prefix_image_sha256=prefix['candidates']['generation_only']['sha256']))
    definition=read_json(root/'data/07_shampoo/latent_only_matched.json')
    definition['pixel_optimization']=True
    definition['initialization']['optimizer'].update(pixel_clean_steps=20,pixel_diffusion_steps=60,pixel_levels=4)
    definition['latent']['auxiliary_levels']=4
    freeze_json(root/'data/07_shampoo/pixel_control_matched.json',definition)
    with catalog_training_guard(allowed_output_roots=[root/'data',destination,parent,
            source/'foreground_segmentation',source/'foreground_statistics']) as audit:
        fork=destination/'initialization/attack/checkpoint.pt'
        if not fork.exists():
            skip={'ready.json','run_identity.json','RESUMED','protocol.json',
                  'attack/checkpoint.pt','attack/identity.json','attack/metadata.json','attack/progress.json'}
            for path in parent.rglob('*'):
                if path.is_file() and path.relative_to(parent).as_posix() not in skip:
                    exact_copy(path,destination/'initialization'/path.relative_to(parent))
            for name in ('foreground_segmentation','foreground_statistics'):
                for path in (source/name).rglob('*'):
                    if path.is_file():exact_copy(path,destination/name/path.relative_to(source/name))
            state=torch.load(parent/'attack/checkpoint.pt',map_location='cpu',weights_only=False)
            if state['phase_index']!=4 or state['phase_step']!=0:
                raise ValueError('公共潜变量初始化没有完整完成')
            zero=torch.zeros_like(state['pixel'])
            if torch.count_nonzero(state['pixel']) or torch.count_nonzero(state['velocity']):
                raise ValueError('公共前缀携带了像素更新状态')
            identity=dict(state['identity'])
            identity['config']=__import__('json').loads(__import__('json').dumps(asdict(initialization_config(definition))))
            identity['sequence']='instruction_edit_then_latent_and_pixel'
            state.update(identity=identity,phase_index=2,phase_step=0,floors=None,best_row=None,latest=None,
                best=state['generative'].clone(),best_latent=state['latent'].clone(),best_pixel=zero.clone(),
                reference_latent=state['latent'].clone(),reference_pixel=zero.clone(),pixel=zero.clone(),velocity=zero.clone())
            freeze_json(destination/'initialization/attack/identity.json',identity)
            fork.parent.mkdir(parents=True,exist_ok=True)
            torch.save(state,fork)
            freeze_json(destination/'COMMON_PREFIX_REUSE.json',dict(
                protocol='paired_counterfactual_from_completed_public_latent_prefix_v1',
                parent_ready=str(parent/'ready.json'),parent_ready_sha256=sha256_file(parent/'ready.json'),
                parent_checkpoint_sha256=sha256_file(parent/'attack/checkpoint.pt'),
                common_prefix_image_sha256=prefix['candidates']['generation_only']['sha256'],
                completed_latent_steps_reused=80,completed_instruction_generation_steps_reused=30,
                fork_phase='pixel_clean',pixel_steps_to_execute=[20,60],final_latent_steps_to_execute=40,
                private_memory_used=False,formal_queries_used=False,not_a_fresh_prefix_generation=True))
        image,metadata=generate_attack(definition['data']['source'],definition['data']['target'],destination,
            method='PerLAGE',perlage_config=definition,device='cuda')
        completed=read_json(destination/'initialization/ready.json')
        if completed['candidates']['generation_only']['sha256']!=prefix['candidates']['generation_only']['sha256']:
            raise ValueError('含像素配对对照没有保留完全相同的潜变量前缀')
        freeze_json(destination/'completion.json',metadata)
    freeze_json(destination/'TRAINING_ACCESS_AUDIT.json',{k:v for k,v in audit.items() if k!='active'})
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'lucid', 'perlage', 'pixel-control', 'evaluate', 'verify', 'all'])
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--epsilon-levels', type=int, choices=[4,8], default=4)
    parser.add_argument('--wait-for-pid',type=int)
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    if args.stage == 'prepare':
        prepare(args.root)
    elif args.stage == 'lucid':
        generate_lucid(args.root, args.epsilon_levels)
    elif args.stage == 'perlage':
        generate_perlage(args.root)
    elif args.stage == 'pixel-control':
        generate_pixel_control(args.root)
    elif args.stage in ('evaluate','verify'):
        from src.experiments import shampoo_ablation_evaluation as evaluation
        getattr(evaluation,args.stage)(args.root)
    else:
        run_queue(args.root,wait_for_pid=args.wait_for_pid)


def run_queue(root,*,wait_for_pid=None):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    logs=root/'logs';logs.mkdir(exist_ok=True)
    if not (root/'HISTORICAL_IMAGES.json').exists():
        prepare(root)
    if wait_for_pid:
        import psutil
        previous=read_json(root/'RUN_STATUS.json') if (root/'RUN_STATUS.json').exists() else {}
        save_json(root/'RUN_STATUS.json',dict(status='running',stage=previous.get('stage','lucid_4'),
            child_pid=wait_for_pid,supervisor_pid=os.getpid(),waiting_for_existing_generation=True))
        while psutil.pid_exists(wait_for_pid):
            time.sleep(5)
        if not (root/'generation/lucid_4/completion.json').is_file():
            raise RuntimeError('前置四级像素优化未完成，队列停止')
    commands=[('lucid_4',['lucid','--epsilon-levels','4']),('lucid_8',['lucid','--epsilon-levels','8']),
              ('perlage_latent_only',['perlage']),('perlage_pixel_control',['pixel-control']),
              ('evaluate',['evaluate']),('verify',['verify'])]
    env=dict(os.environ,PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',
             MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
    for name,arguments in commands:
        marker=(root/'generation'/name/'completion.json' if name.startswith(('lucid_','perlage_'))
                else root/'evaluation'/('SIMILARITIES.json' if name=='evaluate' else 'INDEPENDENT_VERIFICATION.json'))
        if marker.is_file():continue
        state=dict(status='running',stage=name,started_at=datetime.now(timezone.utc).isoformat(),
                   supervisor_pid=os.getpid(),log=str(logs/(name+'.log')))
        save_json(root/'RUN_STATUS.json',state)
        started=time.perf_counter()
        with (logs/(name+'.log')).open('a',encoding='utf-8') as stream:
            process=subprocess.Popen([sys.executable,'-X','utf8','-B','-u','-m','src.experiments.shampoo_ablation',
                *arguments,'--root',str(root)],cwd=PROJECT,env=env,stdout=stream,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            state['child_pid']=process.pid;save_json(root/'RUN_STATUS.json',state)
            code=process.wait()
        state.update(status='completed' if code==0 else 'failed',exit_code=code,seconds=time.perf_counter()-started,
                     finished_at=datetime.now(timezone.utc).isoformat())
        save_json(root/'stages'/(name+'.json'),state)
        if code:
            save_json(root/'RUN_STATUS.json',state)
            raise RuntimeError('阶段失败并停止：'+name)
    save_json(root/'RUN_STATUS.json',dict(status='completed',completed_at=datetime.now(timezone.utc).isoformat(),
        report=str(root/'REPORT.md')))


if __name__ == '__main__':
    main()
