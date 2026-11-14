"""准备9组图对、目标特点文本与完善后的189条全局长期记忆库（含检索向量缓存）。

本脚本将选图产物、目标特点文本与完善后的长期记忆库整合成统一清晰的基础准备资产，
存放于 output/experiment_base_groups9_20261109/ 目录下，
供后续 lucid、PerLAGE 和 PerLAGE_no_pixel 三种方法进行统一对齐训练与检索评测读取。
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import sqlite3

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Windows 控制台 UTF-8 支持
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from src.paths import OUTPUT, SHARED_MEMORY, EXPERIMENT_BASE_GROUPS9
from src.visual.audit import digest, freeze_json, read_json, save_json
from src.visual.image_utils import sha256_file
from src.visual.retrieval_objective import TRAINING_DATA_POLICY
from src.visual.target_features import (
    extract_product_features,
    format_retrieval_text,
    CATALOG_DB_PATH,
    CURATED_TARGET_FEATURES,
)
from src.visual.visual_retrieval import (
    VisualMemoryRetriever,
    memory_index_text,
    resolve_image,
    PRODUCT_FEATURE_INDEX_TEXT_POLICY,
    canonical_digest,
)

SOURCE_REVIEW_DIR = OUTPUT / "public_pair_selection/groups9_review_20261108"
BASE_POOL_PATH = SHARED_MEMORY


def prepare_public_metadata():
    """仅首次只读投影公开字段；后续选图、训练不依赖旧项目输出。"""
    review = EXPERIMENT_BASE_GROUPS9 / 'audit/verified_product_overrides.json'
    local_review = PROJECT_ROOT / 'data/public/product_feature_review.json'
    if not review.is_file():
        review = local_review
    if not review.is_file():
        raise FileNotFoundError('缺少逐字段核对的公开特点审核记录')
    reviewed = read_json(review)
    local_review.parent.mkdir(parents=True, exist_ok=True)
    if review.resolve() != local_review.resolve():
        shutil.copyfile(review, local_review)
    wanted = {e['FinalASIN'] for e in read_json(BASE_POOL_PATH)['entries']} | set(reviewed['products'])
    projection = PROJECT_ROOT / 'data/public/product_metadata.json'
    products = read_json(projection).get('products', {}) if projection.is_file() else {}
    source = Path(reviewed['source_catalog'])
    fields = ('asin','Title','name','Brand','brand','Attributes','BulletPoints','Description',
              'product_category','category','query','MainImage','images')
    missing = wanted - set(products)
    if missing:
        with sqlite3.connect(f'file:{source.as_posix()}?mode=ro', uri=True) as conn:
            for asin in sorted(missing):
                row = conn.execute('SELECT data FROM products WHERE asin=?', (asin,)).fetchone()
                if row is None:
                    raise ValueError(f'公开目录缺少商品 {asin}')
                data = json.loads(row[0])
                products[asin] = {k: data[k] for k in fields if k in data}
    save_json(projection, dict(schema_version=1, source_catalog_sha256=reviewed['source_catalog_sha256'],
                              excluded_fields=['instructions','instruction','actions','purchase_history'], products=products))
    metadata_dir = EXPERIMENT_BASE_GROUPS9 / 'public_metadata'
    metadata_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(projection, metadata_dir / projection.name)
    shutil.copyfile(local_review, metadata_dir / local_review.name)
    return dict(path='public_metadata/product_metadata.json', sha256=sha256_file(projection),
                review_path='public_metadata/product_feature_review.json', review_sha256=sha256_file(local_review),
                products=len(products), training_source=str(projection.resolve()))


def reference_entries():
    """只建立新九组原商品的如实参考，不继承其他商品的购买行为。"""
    result = {}
    for gdir in sorted((SOURCE_REVIEW_DIR / 'groups').iterdir()):
        if not gdir.is_dir():
            continue
        definition = read_json(gdir / 'definition.json')
        source = definition['source']
        feat = extract_product_features(source['id'])
        image = gdir / 'source.jpg'
        result[gdir.name] = dict(
            Id='groups9_reference_' + gdir.name + '_' + source['id'],
            SourceGroup=gdir.name, EntryType='error', RecordKind='simulated_gallery',
            Completed=False, Success=False, Actions=[], Instruction='', FinalASIN=source['id'],
            ObservedPurchase=False, RecordOrigin='user_selected_public_catalog_reference',
            ReferenceNotice='未观察到购买；来自本次已选公开原商品，仅作模拟商品参考。',
            PrimaryImage=dict(ASIN=source['id'], Title=feat['title'],
                              LocalPath=str(image.resolve()), SHA256=sha256_file(image),
                              SourceURL=read_json(CATALOG_DB_PATH)['products'][source['id']].get('MainImage',''),
                              Role='catalog_reference_not_purchase'))
    if len(result) != 9:
        raise ValueError('必须恰好复用九组新图对')
    return result


def enrich_shared_memory(base_path: Path, dest_dir: Path) -> dict:
    """完善189条全局长期记忆库条目：补充品牌、特点，升级检索文本策略并排查操作轨迹。"""
    print("\n[1/3] 正在完善189条全局长期记忆库...")
    dest_dir.mkdir(parents=True, exist_ok=True)
    images_dest = dest_dir / "images"
    images_dest.mkdir(parents=True, exist_ok=True)

    base_pool = read_json(base_path)
    entries = base_pool.get("entries", [])
    if len(entries) != 189:
        raise ValueError(f"预期基础共享库应包含 189 条条目，实际为 {len(entries)} 条")

    enriched_entries = []
    brand_stats = []
    replacements = reference_entries()
    mapping = {}

    for idx, raw_entry in enumerate(entries, start=1):
        entry = copy.deepcopy(raw_entry)
        if entry.get('EntryType') == 'error':
            group = entry['SourceGroup']
            entry = copy.deepcopy(replacements[group])
            mapping[group] = dict(source_asin=entry['FinalASIN'], source_memory_id=entry['Id'])
        asin = entry.get("FinalASIN")
        if not asin:
            raise ValueError(f"条目缺少 FinalASIN: {entry.get('Id')}")

        feat = extract_product_features(asin)
        brand = feat["brand"]
        attributes = feat["attributes"]
        feature_text = feat["canonical_sentence"]
        image = resolve_image(entry, base_path)
        if image is None:
            raise ValueError(f'缺少原始图片: {asin}')
        target_image = images_dest / (sha256_file(image) + image.suffix.lower())
        if not target_image.is_file() or sha256_file(target_image) != sha256_file(image):
            shutil.copyfile(image, target_image)
        entry['PrimaryImage'].update(LocalPath=target_image.relative_to(dest_dir).as_posix(), SHA256=sha256_file(target_image))

        # 完善条目字段
        entry["Brand"] = brand
        entry["Attributes"] = attributes
        entry["FeatureText"] = feature_text
        entry['FeatureProvenance'] = dict(source=feat['source'], brand_evidence=feat['brand_evidence'], feature_evidence=feat['feature_evidence'])

        # 同步完善 PrimaryImage 记录
        if "PrimaryImage" in entry and isinstance(entry["PrimaryImage"], dict):
            entry["PrimaryImage"]["Brand"] = brand
            entry["PrimaryImage"]["Attributes"] = attributes
            entry["PrimaryImage"]["FeatureText"] = feature_text

        # 严正验证：确保操作轨迹 (Actions/PolicyTrace/Reward) 不参与检索文本
        computed_retrieval_text = memory_index_text(entry, mode="product_feature")
        for forbidden in ("Actions and observations", "action: think[", "click[", "Reward", "PolicyTrace"):
            if forbidden.lower() in computed_retrieval_text.lower():
                raise AssertionError(f"检索文本违规包含操作轨迹标记: {forbidden}")

        brand_stats.append(brand)
        enriched_entries.append(entry)

    dest_pool_file = dest_dir / "shared_memory.json"
    new_pool_meta = {**copy.deepcopy(base_pool), **{
        "schema_version": 1,
        "scope": "shared",
        "entries": enriched_entries,
        "protocol": "global_shared_pool_with_product_features_v1",
        "total_entries": len(enriched_entries),
        "normal_entries": sum(1 for e in enriched_entries if e.get("EntryType") == "normal"),
        "error_entries": sum(1 for e in enriched_entries if e.get("EntryType") == "error"),
        "retrieval_policy": PRODUCT_FEATURE_INDEX_TEXT_POLICY,
        "retrieval_fields": ["PrimaryImage.Title", "Brand", "Attributes", "FeatureText"],
        "excluded_from_retrieval": ["Actions", "PolicyTrace", "Reward", "Instruction"],
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "notes": "已完善商品名称、品牌（Brand）、特点（Attributes/FeatureText），严禁操作轨迹进入编码向量",
        'training_usage': 'forbidden',
        'input_sources': [dict(path=str(base_path.resolve()), sha256=sha256_file(base_path))],
        'experiment_source_mapping': mapping,
        'record_kind_counts': {kind: sum(e.get('RecordKind') == kind for e in enriched_entries)
                               for kind in ('purchase_memory','simulated_gallery')},
    }}
    save_json(dest_pool_file, new_pool_meta)
    print(f"已成功写入完善后的长期记忆库: {dest_pool_file} (条目数: {len(enriched_entries)})")

    return {
        "pool_path": str(dest_pool_file),
        "total_entries": len(enriched_entries),
        "brands_covered": len(set(brand_stats)),
        "pool_sha256": sha256_file(dest_pool_file),
    }


def precompute_index_cache(dest_memory_path: Path, encoder_config=None) -> dict:
    """使用 CLIP 真实编码器预计算完善后记忆库的【商品名称+品牌+特点】检索向量缓存。"""
    print("\n[2/3] 正在预计算完善后记忆库的检索向量缓存 (product_feature 策略)...")
    from src.experiments.shared_evaluation import make_eval_encoder

    device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") != "" else "cpu"
    try:
        import torch
        if not torch.cuda.is_available():
            device = "cpu"
    except Exception:
        device = "cpu"

    print(f"载入视觉联合编码器 (设备: {device})...")
    if device != 'cuda':
        raise RuntimeError('本轮准备要求显卡上的正式留出检索器，不静默降级')
    encoder = make_eval_encoder(device=device, config_path=encoder_config)

    retriever = VisualMemoryRetriever(
        encoder=encoder,
        memory_path=dest_memory_path,
        key_mode="joint",
        query_mode="text",
        index_text="product_feature",
        cache_dir="auto",
    )

    entries = read_json(dest_memory_path)["entries"]
    identity = retriever.index_identity(entries)
    scope = canonical_digest(identity)
    cache_path = retriever.cache_dir / f"{scope}.json"

    print(f"开始对 189 条商品逐条进行【图片 + 名称/品牌/特点】联合编码...")
    start_t = time.perf_counter()
    keys = []
    for idx, entry in enumerate(entries, start=1):
        img_path = resolve_image(entry, dest_memory_path)
        if img_path is None:
            raise ValueError(f'预计算缺少图片: {entry["Id"]}')
        text = memory_index_text(entry, "product_feature")
        vector, key = retriever._encode(text, img_path, "joint", scope=scope, image_weight=0.5)
        keys.append(key)
        if idx % 50 == 0 or idx == len(entries):
            print(f"  已编码 {idx}/{len(entries)} 条记忆...")

    retriever._save_index_cache(cache_path, identity, keys)
    if len(keys) != 189:
        raise ValueError('向量缓存必须覆盖全部189条')
    # 新检索器只装载磁盘缓存；若其需要实际重编码，立即报错。
    class CacheOnlyEncoder:
        snapshot = encoder.snapshot
        supports_image_weight = True
        def encode_joint(self, *args, **kwargs):
            raise AssertionError('磁盘缓存未真正命中')
        def encode_text(self, *args, **kwargs):
            raise AssertionError('记忆缓存重载核验不应编码查询')
    loaded = VisualMemoryRetriever(CacheOnlyEncoder(), dest_memory_path, query_mode='text', index_text='product_feature', cache_dir='auto')
    if loaded.index_identity(entries) != identity:
        raise ValueError('重新加载的缓存身份不一致')
    loaded._load_index_cache(cache_path, identity)
    for entry in entries:
        loaded._encode(memory_index_text(entry, 'product_feature'), resolve_image(entry, dest_memory_path),
                       'joint', scope=scope, image_weight=0.5)
    cost = time.perf_counter() - start_t
    print(f"向量缓存已保存至: {cache_path} (耗时: {cost:.2f}s, 包含 {len(keys)} 个向量)")

    result = {
        "cache_path": str(cache_path),
        "vector_count": len(keys),
        "scope": scope,
        "cache_sha256": sha256_file(cache_path),
        'encoder': encoder.snapshot, 'device': str(encoder.device),
        'disk_reload_verified_vectors': len(keys),
    }
    encoder.close()
    return result


def prepare_groups_assets(src_review_dir: Path, dest_dir: Path) -> dict:
    """搬运与升级9组图对、掩码、核对图及统一驱动的 definition.json。"""
    print("\n[3/3] 正在准备9组商品图对与统一文字对齐训练资产...")
    groups_out = dest_dir / "groups"
    groups_out.mkdir(parents=True, exist_ok=True)

    src_groups_dir = src_review_dir / "groups"
    if not src_groups_dir.is_dir():
        raise FileNotFoundError(f"未找到源审核目录: {src_groups_dir}")

    group_manifests = {}

    for gdir in sorted(src_groups_dir.iterdir()):
        if not gdir.is_dir():
            continue
        key = gdir.name
        out_gdir = groups_out / key
        out_gdir.mkdir(parents=True, exist_ok=True)

        # 复制图像与掩码文件
        asset_files = [
            "target.jpg",
            "source.jpg",
            "target_foreground.png",
            "source_foreground.png",
            "review_overlay_target.png",
            "review_overlay_source.png",
            "contact_sheet.jpg",
        ]
        for fname in asset_files:
            src_f = gdir / fname
            dest_f = out_gdir / fname
            if src_f.is_file() and (not dest_f.is_file() or sha256_file(dest_f) != sha256_file(src_f)):
                shutil.copyfile(src_f, dest_f)

        # 读取既有 definition.json
        orig_defn_file = gdir / "definition.json"
        orig_defn = read_json(orig_defn_file) if orig_defn_file.is_file() else {}

        target_id = orig_defn.get("target", {}).get("id")
        source_id = orig_defn.get("source", {}).get("id")
        if not target_id or not source_id:
            raise ValueError(f"组 {key} 缺少 target_id 或 source_id")

        # 提取目标商品与源商品特点
        target_feat = extract_product_features(target_id)
        source_feat = extract_product_features(source_id)
        orig_defn['source']['title'] = source_feat['title']
        orig_defn['target']['title'] = target_feat['title']

        # 写入 features.json
        features_file = out_gdir / "features.json"
        save_json(features_file, {
            "group": key,
            "category": orig_defn.get("category", ""),
            "target": target_feat,
            "source": source_feat,
            "alignment_canonical_text": target_feat["canonical_sentence"],
            "alignment_training_queries": target_feat["training_queries"],
        })
        from src.experiments.formal_queries import build_feature_text_manifest
        query_relative = f'evaluation_inputs/{key}/queries.json'
        save_json(dest_dir / query_relative, build_feature_text_manifest(target_feat))

        # 组装统一的 definition.json（可供三种方法直接加载）
        t_boxes = orig_defn.get("target", {}).get("boxes", [])
        t_text = orig_defn.get("target", {}).get("text_regions", [])
        s_boxes = orig_defn.get("source", {}).get("boxes", [])
        s_text = orig_defn.get("source", {}).get("text_regions", [])

        unified_defn = {
            "schema_version": 2,
            "group": key,
            "name": orig_defn.get("name", key),
            "category": orig_defn.get("category", ""),
            "shape_similarity": orig_defn.get("shape_similarity", 0.0),
            "calibrated_threshold": 0.6010810730721305,
            "passes_calibrated_threshold": orig_defn.get("passes_calibrated_threshold", True),
            "selection_status": orig_defn.get("selection_status", ""),
            "selection_notes": orig_defn.get("selection_notes", ""),
            "methods_supported": ["lucid", "PerLAGE", "PerLAGE_no_pixel"],
            "target_feature_text": target_feat["canonical_sentence"],
            "training_queries": target_feat["training_queries"],
            "data": {
                "source": "source.jpg",
                "target": "target.jpg",
                "source_title": orig_defn.get("source", {}).get("title", ""),
                "training_queries": [target_feat["canonical_sentence"]],
            },
            "source_boxes": s_boxes,
            "target_boxes": t_boxes,
            "print_regions": s_text,
            "target": {
                "id": target_id,
                "title": orig_defn.get("target", {}).get("title", ""),
                "brand": target_feat["brand"],
                "attributes": target_feat["attributes"],
                "feature_text": target_feat["canonical_sentence"],
                "training_queries": target_feat["training_queries"],
                "image": "target.jpg",
                "foreground_mask": "target_foreground.png",
                "boxes": t_boxes,
                "text_regions": t_text,
            },
            "source": {
                "id": source_id,
                "title": orig_defn.get("source", {}).get("title", ""),
                "brand": source_feat["brand"],
                "attributes": source_feat["attributes"],
                "image": "source.jpg",
                "foreground_mask": "source_foreground.png",
                "boxes": s_boxes,
                "text_regions": s_text,
            },
            "color_scheme": "statistics",
            "training_data_policy": TRAINING_DATA_POLICY,
            "lucid_alignment_config": {
                "enabled": True,
                "target_text": target_feat["canonical_sentence"],
                "text_weight": 1.0,
                "steps": 1000,
            },
            "perlage_alignment_config": {
                "enabled": True,
                "training_queries": [target_feat["canonical_sentence"]],
                "color_scheme": "statistics",
                "pixel_optimization": True,
            },
            "perlage_no_pixel_alignment_config": {
                "enabled": True,
                "training_queries": [target_feat["canonical_sentence"]],
                "color_scheme": "statistics",
                "pixel_optimization": False,
            },
        }

        save_json(out_gdir / "definition.json", unified_defn)

        # 生成三种方法立即可读运行的独立配置
        from src.visual._perlage.evidence import DEFINITION as PERLAGE_DEFINITION
        from src.visual._perlage_no_pixel.evidence import DEFINITION as NO_PIXEL_DEFINITION
        save_json(out_gdir / "perlage_config.json", {
            "schema_version": 2,
            "method": "PerLAGE",
            "definition": PERLAGE_DEFINITION,
            "color_scheme": "statistics",
            "training_data_policy": TRAINING_DATA_POLICY,
            "pixel_optimization": True,
            "target_feature_text": target_feat["canonical_sentence"],
            "data": {
                "source": "source.jpg",
                "target": "target.jpg",
                "source_title": orig_defn.get("source", {}).get("title", ""),
                "training_queries": [target_feat["canonical_sentence"]],
            },
            "source_boxes": s_boxes,
            "target_boxes": t_boxes,
            "print_regions": s_text,
        })

        save_json(out_gdir / "perlage_no_pixel_config.json", {
            "schema_version": 2,
            "method": "PerLAGE_no_pixel",
            "definition": NO_PIXEL_DEFINITION,
            "color_scheme": "statistics",
            "training_data_policy": TRAINING_DATA_POLICY,
            "pixel_optimization": False,
            "target_feature_text": target_feat["canonical_sentence"],
            "data": {
                "source": "source.jpg",
                "target": "target.jpg",
                "source_title": orig_defn.get("source", {}).get("title", ""),
                "training_queries": [target_feat["canonical_sentence"]],
            },
            "source_boxes": s_boxes,
            "target_boxes": t_boxes,
            "print_regions": s_text,
        })

        save_json(out_gdir / "lucid_config.json", {
            "schema_version": 2,
            "method": "lucid",
            "target_text": target_feat["canonical_sentence"],
            "text_weight": 1.0,
            "steps": 1000,
            "epsilon_levels": 8,
            "source": "source.jpg",
            "target": "target.jpg",
        })

        group_manifests[key] = {
            "name": unified_defn["name"],
            "category": unified_defn["category"],
            "target_id": target_id,
            "target_brand": target_feat["brand"],
            "target_feature_text": target_feat["canonical_sentence"],
            "source_id": source_id,
            "shape_similarity": unified_defn["shape_similarity"],
            'passes_calibrated_threshold': unified_defn['passes_calibrated_threshold'],
            'selection_basis': 'user_confirmed_existing_nine_pairs_no_reselection',
            'source_sha256': sha256_file(out_gdir / 'source.jpg'),
            'target_sha256': sha256_file(out_gdir / 'target.jpg'),
            "directory": f"groups/{key}",
            "definition": f"groups/{key}/definition.json",
            "features": f"groups/{key}/features.json",
            'query_manifest': query_relative,
            'query_manifest_sha256': sha256_file(dest_dir / query_relative),
            'features_sha256': sha256_file(features_file),
            "perlage_config": f"groups/{key}/perlage_config.json",
            "perlage_no_pixel_config": f"groups/{key}/perlage_no_pixel_config.json",
            "lucid_config": f"groups/{key}/lucid_config.json",
            'method_config_sha256': {method: sha256_file(out_gdir / (method + '_config.json'))
                                     for method in ('lucid','perlage','perlage_no_pixel')},
        }
        print(f"  [{key}] 已完成特点提取与统一配置: {target_feat['canonical_sentence']}")

    write_corrected_previews(dest_dir, group_manifests)

    return group_manifests


def write_corrected_previews(dest_dir, groups):
    """从真实原图及已核对标题重建预览；不沿用昨日错误商品文案。"""
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    import textwrap
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 18)
    small = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 16)
    sheets = []
    for key, meta in groups.items():
        directory = dest_dir / meta['directory']
        definition = read_json(directory / 'definition.json')
        canvas = Image.new('RGB', (800, 650), 'white')
        draw = ImageDraw.Draw(canvas)
        draw.text((15, 10), definition['name'] + ' | ' + key, font=font, fill='black')
        for col, role, label in [(0, 'source', '原商品'), (1, 'target', '目标商品')]:
            record = definition[role]
            x = 10 + col * 400
            draw.text((x, 45), label + ' ' + record['id'], font=font, fill='black')
            with Image.open(directory / record['image']) as image:
                thumb = ImageOps.contain(image.convert('RGB'), (380, 310))
            canvas.paste(thumb, (x+(380-thumb.width)//2, 78+(310-thumb.height)//2))
            for n, line in enumerate(textwrap.wrap(record['title'], width=44)[:7]):
                draw.text((x, 400+n*19), line, font=small, fill='black')
        feature = definition['target_feature_text']
        for n,line in enumerate(textwrap.wrap(feature, width=92)):
            draw.text((15, 548+n*20), line, font=small, fill='black')
        gate = '历史门槛通过' if definition['passes_calibrated_threshold'] else '历史门槛未通过；用户指定复用'
        draw.text((15, 615), gate + '；本轮不重新选图', font=font, fill='#333333')
        canvas.save(directory / 'contact_sheet.jpg', quality=93)
        sheets.append(canvas)
    overview = Image.new('RGB', (2400, 1950), '#dddddd')
    for n, sheet in enumerate(sheets):
        overview.paste(sheet, ((n%3)*800, (n//3)*650))
    overview.save(dest_dir / 'CONTACT_SHEET_ALL9.jpg', quality=93)


def build_experiment_base_package():
    """主编排入口：构建统一实验基础资产目录。"""
    out_dir = EXPERIMENT_BASE_GROUPS9
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = prepare_public_metadata()
    print(f"==================================================================")
    print(f"开始构建 9 组实验基础准备包: {out_dir}")
    print(f"==================================================================")

    # 1. 完善长期记忆库
    shared_mem_meta = enrich_shared_memory(BASE_POOL_PATH, out_dir / "shared_memory")

    # 桌面待执行任务显式指定此模型；它属于训练代理，不能宣称独立留出。
    encoder_config = out_dir / 'evaluation_encoder.json'
    spec = read_json(PROJECT_ROOT / 'src/visual/config/optimization.json')['surrogates'][1]
    save_json(encoder_config, dict(setting='matched_public_encoder', surrogate=spec,
        reason='user_desktop_task_explicit_ViT_B_32', black_box_transfer_claimed=False))
    # 2. 预计算同一模型、同一查询模式下的完整记忆键缓存
    cache_meta = precompute_index_cache(Path(shared_mem_meta["pool_path"]), encoder_config=encoder_config)

    # 3. 准备 9 组图对与统一文字对齐训练配置
    groups_manifest = prepare_groups_assets(SOURCE_REVIEW_DIR, out_dir)

    # 4. 生成根索引清单 manifest.json
    manifest = {
        "schema_version": 1,
        "title": "9组商品图对、特点文字与完善长期记忆库的统一实验准备资产",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_base": str(out_dir),
        'public_metadata': metadata,
        'evaluation_protocol': dict(query_mode='text', query_images=False, queries_per_group=20,
            instruction_policy='public_target_features_fixed20_v1', conditions=['none'],
            encoder_config='evaluation_encoder.json', encoder_config_sha256=sha256_file(encoder_config),
            evaluator_setting='matched_public_encoder', black_box_transfer_claimed=False,
            shared_pool_scope='all_189_entries', final_shopping_enabled=False),
        "shared_memory": {
            "pool_file": "shared_memory/shared_memory.json",
            "pool_sha256": shared_mem_meta["pool_sha256"],
            "total_entries": shared_mem_meta["total_entries"],
            'source_mapping': read_json(shared_mem_meta['pool_path'])['experiment_source_mapping'],
            'experiment_sources': 'nine_new_public_source_references_not_observed_purchases',
            "brands_covered": shared_mem_meta["brands_covered"],
            "retrieval_index_policy": "PRODUCT_FEATURE_INDEX_TEXT_POLICY (Title + Brand + Features, excluding actions)",
            "index_cache": {
                "cache_file": f"shared_memory/retrieval_index_cache/{cache_meta['scope']}.json",
                "cache_sha256": cache_meta["cache_sha256"],
                "vector_count": cache_meta["vector_count"],
                'encoder': cache_meta['encoder'], 'device': cache_meta['device'],
                'disk_reload_verified_vectors': cache_meta['disk_reload_verified_vectors'],
            },
        },
        "methods_unified": {
            "lucid": "默认启用商品特点文字对齐 (target_text=feature_text, text_weight=1.0)",
            "PerLAGE": "四阶段完整自然优先生成，训练请求统一使用商品特点文字",
            "PerLAGE_no_pixel": "零像素扰动完整自然优先生成，训练请求统一使用商品特点文字",
        },
        "groups": groups_manifest,
        "ready_for_generation_and_evaluation": True,
        'generated_attack_images_complete': False,
        'formal_evaluation_performed': False,
    }

    save_json(out_dir / "manifest.json", manifest)
    print(f"\n==================================================================")
    print(f"实验基础准备包已构建完成！")
    print(f"根清单文件: {out_dir / 'manifest.json'}")
    print(f"包含 9 组完整图对资产与特点文本，以及 189 条已重新编码的长期记忆库。")
    print(f"==================================================================")


if __name__ == "__main__":
    build_experiment_base_package()
