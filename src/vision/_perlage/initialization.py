from src.paths import model_path
"""无影子池的独立生成起点；完整重新生成，不导入有池训练资产。"""
from src.paths import relocate_io_path
from dataclasses import asdict
import gc, hashlib, json, time
from src.paths import ProjectPath as Path, PROJECT
import numpy as np
import torch
from src.vision._natural.optimizer import InitializationOptimizationConfig, refine_generation_initialization
from src.vision._natural.assets import NaturalGenerationConfig, read_json, prepare_generation_anchor
from src.vision.natural_visual_edit import EditPlan
from src.vision._natural.editor import SurfaceInstructionEditor
from src.vision._natural.mask import SurfaceMaskConfig, build_surface_masks
from src.vision.image_utils import sha256_file, write_json
from src.vision.retrieval_objective import TRAINING_DATA_POLICY, validate_public_title

def prepare_generation_initialization(source, target, output, *, plan: EditPlan, source_title, training_queries, surface_regions, surface_mask_config=SurfaceMaskConfig(), config=InitializationOptimizationConfig(shadow_weight=0.), generation=NaturalGenerationConfig(), base_model=model_path('stable-diffusion-v1-5'), editor_model=model_path('instruct-pix2pix-editor'), device='cuda', pause_path=None):
    """内部初始化计算；源图可改、目标图只读，完成后仍须执行完整方法。"""
    config.validate()
    if config.shadow_weight != 0:
        raise ValueError("无影子池初始化不允许启用竞争池损失")
    source_title = validate_public_title(source_title)
    surface_mask_config.validate()
    source, target = (Path(source).resolve(), Path(target).resolve())
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    binding = dict(shadow_pool_used=False, preparation_seconds=0., selection="unthresholded_owned_query_scores")
    identity = dict(method='perlage_generation_initialization', source=str(source), target=str(target), inputs={'source': sha256_file(source), 'target': sha256_file(target)}, source_title_sha256=hashlib.sha256(source_title.encode('utf-8')).hexdigest(), training_queries=list(training_queries), plan=asdict(plan), config=asdict(config), generation=asdict(generation), surface_regions=surface_regions, surface_mask_config=asdict(surface_mask_config), public_binding=binding, code={name: sha256_file(PROJECT / name) for name in ['src/vision/_perlage/initialization.py', 'src/vision/_natural/editor.py', 'src/vision/_natural/mask.py', 'src/vision/_natural/optimizer.py', 'src/vision/_natural/core.py', 'src/vision/_natural/assets.py', 'src/vision/natural_visual_edit.py']}, base_model=str(Path(base_model).resolve()), editor_model=str(Path(editor_model).resolve()), device=device, private_gallery_used=False, true_rank_used=False, purchase_runs=0)
    identity.update(training_data_policy=TRAINING_DATA_POLICY, purchase_history_used=False, victim_memory_used=False)
    identity = json.loads(json.dumps(identity))
    ip = root / 'run_identity.json'
    if ip.exists() and read_json(ip) != identity:
        raise ValueError('独立方法输入或实现改变，请使用新目录')
    write_json(ip, identity)
    if (root / 'ready.json').exists():
        ready = read_json(root / 'ready.json')
        if ready['run_identity_sha256'] != sha256_file(ip):
            raise ValueError('完成记录身份改变')
        if any((sha256_file(v['image']) != v['sha256'] for v in ready['candidates'].values())):
            raise ValueError('完成图片改变')
        if sha256_file(ready['mask_path']) != ready['generation_mask_sha256']:
            raise ValueError('生成保护掩码改变')
        if ready['content_mask_record_sha256'] != sha256_file(ready['content_mask_record']):
            raise ValueError('内容掩码记录改变')
        saved_mask_record = read_json(ready['content_mask_record'])
        if any((sha256_file(v['path']) != v['sha256'] for v in saved_mask_record['files'].values())):
            raise ValueError('内容保护掩码改变')
        return (Path(ready['candidates']['natural_final']['image']), ready)
    pause_path = Path(pause_path) if pause_path else root / 'PAUSE_REQUESTED'
    if pause_path.exists():
        raise InterruptedError('独立方法保持暂停')
    resumed = any(((root / x).exists() for x in ['attack/checkpoint.pt', 'generation/checkpoint.pt']))
    if resumed:
        write_json(root / 'RESUMED', dict(reason='existing_generation_or_optimizer_checkpoint'))
    from src.vision.dual_rag_bridge import make_encoder
    from src.vision.joint_encoder import ChunkedClipEncoder
    from src.vision.diffusion_purification import LatentDiffusionPurifier, PurificationConfig
    started = time.perf_counter()
    mask_started = time.perf_counter()
    mask_record = build_surface_masks(source, surface_regions, root / 'content_masks', surface_mask_config)
    mask_seconds = time.perf_counter() - mask_started
    base = make_encoder(device=device)
    encoder = ChunkedClipEncoder(base)
    own = encoder.encode_text(source_title)
    queries = np.stack([encoder.encode_joint(q, str(target)) for q in training_queries])
    editor = SurfaceInstructionEditor(base_model, editor_model, base, device=device)
    settings = asdict(generation)
    seed = settings.pop('seed')
    generated, gen = editor.edit(source, target, root / 'generation', plan, seed=seed, target_guidance=True, title_feature=own, query_features=queries, edit_mask_path=mask_record['files']['edit']['path'], **settings)
    del editor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    anchor = prepare_generation_anchor(source, generated, root / 'initialization', device=base.surrogate.device)
    purifier = LatentDiffusionPurifier(PurificationConfig(working_size=512), device=device)
    purifier.vae.enable_gradient_checkpointing()
    bank = None
    image, meta = refine_generation_initialization(source, target, anchor, root / 'generation/edit_mask.png', root / 'attack', surrogate=base.surrogate, source_title=source_title, training_queries=training_queries, purifier=purifier, public_bank=bank, config=config, pause_path=pause_path)
    if meta['status'] != 'completed':
        raise InterruptedError('独立方法已保存暂停断点')
    candidates = {name: dict(image=str(path.resolve()), sha256=sha256_file(path)) for name, path in [('generation_only', root / 'attack/generative_anchor.png'), ('natural_final', image)]}
    ready = dict(status='completed', method='perlage_generation_initialization', candidates=candidates, run_identity_sha256=sha256_file(ip), encoder=base.snapshot, purifier=purifier.snapshot(), generation_seconds=gen['generation_seconds'], refinement_and_development_seconds=meta['elapsed_seconds'], necessary_optimization_development_seconds=gen['generation_seconds'] + meta['elapsed_seconds'] + mask_seconds, content_mask_preparation_seconds=mask_seconds, content_mask_record=str((root / 'content_masks/mask_record.json').resolve()), content_mask_record_sha256=sha256_file(root / 'content_masks/mask_record.json'), fine_ink_protection=True, content_semantics_certified=False, requires_manual_content_and_logo_review=True, observed_call_wall_seconds=time.perf_counter() - started, full_fresh_call_wall_observed=not (root / 'RESUMED').exists(), public_bank_preparation_seconds=binding['preparation_seconds'], public_bank_reused=False, shadow_pool_used=False, mask_path=str((root / 'generation/edit_mask.png').resolve()), generation_mask_sha256=sha256_file(root / 'generation/edit_mask.png'), hardware=torch.cuda.get_device_name(0) if device.startswith('cuda') else 'cpu', private_gallery_used=False, true_rank_used=False, purchase_runs=0)
    if (root / 'protocol.json').exists():
        ready['protocol_sha256'] = sha256_file(root / 'protocol.json')
    write_json(root / 'ready.json', ready)
    del base, encoder, purifier, bank
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return (image, ready)
