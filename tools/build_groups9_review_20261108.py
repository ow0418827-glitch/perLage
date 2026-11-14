"""为9组图对生成审核包：原图、目标图、掩码标注、单组核对图及全量9组总览图。"""
import os
import sys
from pathlib import Path

# 确保以 UTF-8 编码输出
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import json
import shutil
import sqlite3
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.visual.audit import freeze_json, save_json, read_json, digest
from src.visual.image_utils import sha256_file
from src.visual.public_annotations import PublicImageAnnotator
from src.visual.pair_selection import LocalSelectionBackend, unit_rows
from src.visual.selection.webshop_public import download_public_image
from src.visual.target_features import extract_product_features

OUTPUT_BASE = PROJECT_ROOT / 'output/public_pair_selection/groups9_review_20261108'
DB_PATH = PROJECT_ROOT / 'output/datasets/public/public_products_v2.sqlite'
THRESHOLD_PATH = PROJECT_ROOT / 'output/datasets/public/calibration/shape_v2_verified/threshold.json'

GROUPS_SPEC = [
    {
        'key': '01_camera',
        'name': '第一组相机',
        'category': 'camera',
        'target_id': 'B08V134QWR',
        'target_title': 'PTZ Security Camera Outdoor 30X Optical Zoom Camera WiFi Wireless 5MP RTSP IP Camera 1000FT Night Vision Auto Tracking Emergency Strobe Light Alarm Human Detection 2-Way Audio IP66 Waterproof',
        'source_id': 'B071ZFHTZN',
        'source_title': 'SUNBA 3MP Starlight IP PoE+ Outdoor PTZ Camera 20x Optical Zoom, Auto Tracking & Humanoid Detection, Night Vision up to 1000ft, RTMP/RTSP 1080P/3MP 60FPS Speed Dome (Illuminati-20X)',
        'shape_similarity': 0.7198,
        'selection_status': 'screened_high_similarity',
        'selection_notes': '通过公开粗类型兼容性门槛与灰度外形特征门槛（0.7198 >= 0.601081）。受限于粗筛预算hash截断，本组系在模拟可控公开池中筛查出的高质量PTZ监控球机同构对。'
    },
    {
        'key': '02_chair',
        'name': '第二组椅子',
        'category': 'seating',
        'target_id': 'B09CYFD5BX',
        'target_title': 'LTHDD Barber Chair Heavy Duty Hydraulic Reclining Styling Salon Chair Shampoo Chair 360 Degree Swivel Vintage Classic for Hair Stylist Beauty Equipment',
        'source_id': 'B08GCSRXKS',
        'source_title': 'Buy-Rite Chelsea Styling Chair for Salon, Heavy-Duty Round Chrome Base with T-Footrest & Steel Frame Structure, Premium Vinyl Upholstery with High-Density Foam (Grey/Chrome)',
        'shape_similarity': 0.7436,
        'selection_status': 'strictly_selected_uniform_draw',
        'selection_notes': '现行第二版门槛抽样方案（public_shape_gate_uniform_v2）在seed 42下从4个合格理发椅候选均匀抽样选出（抽中率 1/4）。'
    },
    {
        'key': '03_cookies',
        'name': '第三组饼干',
        'category': 'cookies',
        'target_id': 'B08Z8MHWVS',
        'target_title': 'Eggless Raw Edible Cookie Dough - by Doughlish - Safe to eat at Room Temp - Ready to Eat with Spoon Included in Lid - No Refrigeration Needed - Resealable Tub (Chocolate Chip, 14 Ounce (Pack of 1))',
        'source_id': 'B07BBLWTLM',
        'source_title': 'Natural Decadence, Cookie Dough Chocolate Chip, 14 Ounce',
        'shape_similarity': 0.7180,
        'selection_status': 'strictly_selected_uniform_draw',
        'selection_notes': '现行第二版门槛抽样方案在seed 42、预算256下从3个合格曲奇面团圆桶候选均匀抽样选出（抽中率 1/3）。'
    },
    {
        'key': '04_curtain',
        'name': '第四组窗帘',
        'category': 'curtains',
        'target_id': 'B086PW7TWX',
        'target_title': 'Windows Treatment Blackout Curtains Grommet Drapes for Boys/Girls Kids Bedroom DRINKES ARE ON ME Vintage Wood Board Texture Printed Room Darkening Curtains for Livingroom/Office, 1 Panel 52x45in',
        'source_id': 'B09PHLS2QD',
        'source_title': 'Window Treatment Semi-Sheer Curtain Valance for Living Room, Bedroom, Kitchen, Bathroom, Rustic Wood Board Texture Printed Valances Rod Pocket Small Windows Curtains, 54x18 Inch, 1 Panel',
        'shape_similarity': 0.5703,
        'selection_status': 'screened_high_similarity',
        'selection_notes': '目标为木质纹理印花单片窗帘。在受控公开池木纹窗帘中筛选出最高外形相似度源商品（0.5703）。短名单未达0.601081硬门槛，保留候选供人工评估。'
    },
    {
        'key': '05_hair_extension',
        'name': '第五组接发',
        'category': 'hair_extensions',
        'target_id': 'B01460C2I2',
        'target_title': 'S-noilite Human Hair Extensions 8pcs 18clips 65g-120g Thick Double Weft Clip in Hair Extensions for Full Head Silk Straight Brazilian Remy Hair (#1B Natural Black, 18 Inch 100g)',
        'source_id': 'B07Z7MQSM1',
        'source_title': 'Kinky Straight Clip In Human Hair Extensions for Black Women 8pcs 18 Clips 120g Thick Double Weft Clip In Hair Extension 100% Brazilian Virgin Hair Natural Black (16 inch)',
        'shape_similarity': 0.7229,
        'selection_status': 'strictly_selected_uniform_draw',
        'selection_notes': '现行第二版门槛抽样方案在seed 42下从7个合格接发候选均匀抽样选出（抽中率 1/7）。'
    },
    {
        'key': '06_jeans',
        'name': '第六组牛仔裤',
        'category': 'apparel',
        'target_id': 'B09HH96NJY',
        'target_title': 'Womens High Rise Jeans Stretch Destroyed Bell Bottom Denim Pants Flare Jeans',
        'source_id': 'B07FRKQNFV',
        'source_title': 'Cinch Men\'s Bronze Label Tapered Slim Fit Jeans',
        'shape_similarity': 0.7325,
        'selection_status': 'strictly_selected_uniform_draw',
        'selection_notes': '现行第二版门槛抽样方案在seed 42下从3个合格牛仔长裤候选均匀抽样选出（抽中率 1/3）。'
    },
    {
        'key': '07_shampoo',
        'name': '第七组洗发水',
        'category': 'personal_care',
        'target_id': 'B00BDJODWI',
        'target_title': 'Aveda Invati Advanced Exfoliating Shampoo Rich - 6.7 oz',
        'source_id': 'B08XYBS678',
        'source_title': 'Dove Nourishing Foaming Hand Wash For Clean and Softer Hands Coconut and Almond Milk Cleanser That Washes Away Dirt and Germs 10.1 oz 4 Count',
        'shape_similarity': 0.6185,
        'selection_status': 'frozen_historical_reuse',
        'selection_notes': '按公子明确指示：直接复用既有洗发水图对（多芬洗手液作为源商品，艾凡达洗发水作为目标商品），不重新选图。'
    },
    {
        'key': '08_speaker',
        'name': '第八组音箱',
        'category': 'audio',
        'target_id': 'B01N47U1VI',
        'target_title': 'BuyTVProducts4Less Touch of ECO Solar/Battery Powered 2x3W Bluetooth Speaker Splashproof Portable Wireless MP3 Player with Built-in Micphone, FM Radio/MicroSD Card (Red)',
        'source_id': 'B01590FHS0',
        'source_title': 'Kicker KB6000 2-Way Full Range High-Efficiency Indoor Outdoor Marine Patio Bookshelf Surface Mount Speakers (Pair) | 6.5 inch Woofers, 2x5 inch Horn Tweeters | Quick Mounting System Included',
        'shape_similarity': 0.4333,
        'selection_status': 'screened_high_similarity',
        'selection_notes': '目标为带有细杆的太阳能草坪音箱。受控公开池中通用音箱外形差异大，短名单最高相似度为0.4333（另提供便携音箱B093PR8QZJ供比对）。未达0.601081门槛，供人工核查。'
    },
    {
        'key': '09_toothbrush',
        'name': '第九组牙刷',
        'category': 'oral_care',
        'target_id': 'B09J81RMLD',
        'target_title': 'Baby U Shaped Toothbrush Toddler Silicone Toothbrushes 2-6 Years with 360 Degree Whole Mouth Design for Kids (Blue)',
        'source_id': 'B09PNNXYJ4',
        'source_title': 'Tthxqing U shaped Toothbrush, Food Grade Soft Silicone Brush Head, 360° Oral Teeth Cleaning Design for Toddlers and Kids (Pink 2-6 years)',
        'shape_similarity': 0.6120,
        'selection_status': 'strictly_selected_uniform_draw',
        'selection_notes': '现行第二版门槛抽样方案在seed 42下从3个合格U型牙刷候选均匀抽样选出（抽中率 1/3）。'
    }
]


def load_font(size):
    for font_path in [
        'C:/Windows/Fonts/msyh.ttc',
        'C:/Windows/Fonts/simsun.ttc',
        'C:/Windows/Fonts/arial.ttf'
    ]:
        if Path(font_path).is_file():
            try:
                return ImageFont.truetype(font_path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def ensure_image(product_id, dest_path, db):
    dest = Path(dest_path)
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    row = db.execute('SELECT id, title, image_url, source_url FROM products WHERE id=?', (product_id,)).fetchone()
    if not row:
        raise ValueError(f"公开商品库缺少记录：{product_id}")
    prod = dict(id=row[0], title=row[1], image_url=row[2], source_url=row[3])
    cache = OUTPUT_BASE / 'cache'
    cache.mkdir(parents=True, exist_ok=True)
    rec = download_public_image(prod, cache)
    shutil.copyfile(rec['image'], dest)
    return dest


def build_review_package(force_reannotate=False):
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    groups_dir = OUTPUT_BASE / 'groups'
    groups_dir.mkdir(parents=True, exist_ok=True)

    annotator = None
    db = sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True)

    # 准备字体
    font_title = load_font(18)
    font_meta = load_font(14)
    font_badge = load_font(12)

    group_results = []
    contact_sheet_items = []

    try:
        for idx, spec in enumerate(GROUPS_SPEC, start=1):
            key = spec['key']
            print(f"\n==================== 处理第 {idx}/9 组: {spec['name']} ({key}) ====================")
            gdir = groups_dir / key
            gdir.mkdir(parents=True, exist_ok=True)

            target_img_path = gdir / 'target.jpg'
            source_img_path = gdir / 'source.jpg'
            target_fg = gdir / 'target_foreground.png'
            source_fg = gdir / 'source_foreground.png'
            target_overlay = gdir / 'review_overlay_target.png'
            source_overlay = gdir / 'review_overlay_source.png'
            defn_path = gdir / 'definition.json'

            # 1. 准备目标图与原图（完全自包含于本项目，不依赖任何外部临时目录）
            if key == '07_shampoo':
                shampoo_case = PROJECT_ROOT / 'output/color_revision_20261105/public_cases/07_shampoo'
                if not target_img_path.is_file():
                    shutil.copyfile(shampoo_case / 'target.jpg', target_img_path)
                if not source_img_path.is_file():
                    shutil.copyfile(shampoo_case / 'source.jpg', source_img_path)
                if not target_fg.is_file():
                    shutil.copyfile(shampoo_case / 'target_foreground.png', target_fg)
                if not source_fg.is_file():
                    shutil.copyfile(shampoo_case / 'source_foreground.png', source_fg)
            else:
                ensure_image(spec['target_id'], target_img_path, db)
                ensure_image(spec['source_id'], source_img_path, db)

            # 2. 标注处理（若已有完整掩码及覆盖预览图，直接复用既有标注，否则调用 SAM2/RapidOCR）
            has_existing_annotations = (
                not force_reannotate and
                target_fg.is_file() and source_fg.is_file() and
                target_overlay.is_file() and source_overlay.is_file() and
                defn_path.is_file()
            )
            if has_existing_annotations:
                print("复用既有主体与文字区域标注...")
                prev_defn = read_json(defn_path)
                t_boxes = prev_defn['target'].get('boxes', [])
                t_text = prev_defn['target'].get('text_regions', [])
                s_boxes = prev_defn['source'].get('boxes', [])
                s_text = prev_defn['source'].get('text_regions', [])
            else:
                if annotator is None:
                    print('正在初始化视觉标注器（SAM2 分割 + RapidOCR 文字区域检测）...')
                    annotator = PublicImageAnnotator()
                print(f"正在进行主体与文字区域标注...")
                t_prop = annotator.propose(
                    {'id': spec['target_id'], 'title': spec['target_title'], 'image': str(target_img_path),
                     'image_sha256': sha256_file(target_img_path)},
                    gdir / 'annotations/target'
                )
                s_prop = annotator.propose(
                    {'id': spec['source_id'], 'title': spec['source_title'], 'image': str(source_img_path),
                     'image_sha256': sha256_file(source_img_path)},
                    gdir / 'annotations/source'
                )
                if key != '07_shampoo':
                    shutil.copyfile(t_prop['foreground_mask'], target_fg)
                    shutil.copyfile(s_prop['foreground_mask'], source_fg)
                shutil.copyfile(t_prop['review_overlay'], target_overlay)
                shutil.copyfile(s_prop['review_overlay'], source_overlay)
                t_boxes = t_prop['boxes']
                t_text = t_prop['text_regions']
                s_boxes = s_prop['boxes']
                s_text = s_prop['text_regions']

            # 3. 创建单组核对图 (contact_sheet.jpg, 左右排列 Target vs Source)
            w_box, h_box = 440, 440
            sheet_w = 920
            sheet_h = 580
            sheet = Image.new('RGB', (sheet_w, sheet_h), '#f4f5f7')
            draw = ImageDraw.Draw(sheet)

            # 组头说明
            status_text = {
                'strictly_selected_uniform_draw': '【标准抽样入选】门槛内均匀抽中',
                'frozen_historical_reuse': '【约定直接复用】多芬+艾凡达既有图对',
                'screened_high_similarity': '【高相似候选】受控库筛选同构商品'
            }.get(spec['selection_status'], spec['selection_status'])

            header_text = f"组别: {spec['name']} ({key}) | 外形相似度: {spec['shape_similarity']:.4f} | 状态: {status_text}"
            draw.text((20, 15), header_text, fill='#111827', font=font_title)

            # 左图：目标商品 Target
            t_overlay = Image.open(target_overlay).convert('RGB')
            t_overlay.thumbnail((w_box, h_box))
            x_t = 20 + (w_box - t_overlay.width) // 2
            y_t = 55 + (h_box - t_overlay.height) // 2
            sheet.paste(t_overlay, (x_t, y_t))
            draw.text((20, 505), f"目标商品 (TARGET): {spec['target_id']}", fill='#b91c1c', font=font_meta)
            draw.text((20, 528), spec['target_title'][:55] + '...', fill='#4b5563', font=font_badge)

            # 右图：源商品 Source
            s_overlay = Image.open(source_overlay).convert('RGB')
            s_overlay.thumbnail((w_box, h_box))
            x_s = 480 + (w_box - s_overlay.width) // 2
            y_s = 55 + (h_box - s_overlay.height) // 2
            sheet.paste(s_overlay, (x_s, y_s))
            draw.text((480, 505), f"源商品 (SOURCE): {spec['source_id']}", fill='#15803d', font=font_meta)
            draw.text((480, 528), spec['source_title'][:55] + '...', fill='#4b5563', font=font_badge)

            sheet_path = gdir / 'contact_sheet.jpg'
            sheet.save(sheet_path, quality=95)
            print(f"已生成单组核对图: {sheet_path.name}")

            # 4. 提取目标商品核心特点并写入单组定义与元数据
            target_feat = extract_product_features(spec['target_id'])
            features_path = gdir / 'features.json'
            save_json(features_path, target_feat)
            print(f"已提取目标商品特点文本: {target_feat['canonical_sentence']}")

            meta = {
                'schema_version': 2,
                'group': key,
                'name': spec['name'],
                'category': spec['category'],
                'shape_similarity': spec['shape_similarity'],
                'calibrated_threshold': 0.6010810730721305,
                'passes_calibrated_threshold': spec['shape_similarity'] >= 0.6010810730721305,
                'selection_status': spec['selection_status'],
                'selection_notes': spec['selection_notes'],
                'target_feature_text': target_feat['canonical_sentence'],
                'training_queries': target_feat['training_queries'],
                'target': {
                    'id': spec['target_id'],
                    'title': spec['target_title'],
                    'brand': target_feat['brand'],
                    'attributes': target_feat['attributes'],
                    'feature_text': target_feat['canonical_sentence'],
                    'training_queries': target_feat['training_queries'],
                    'image': 'target.jpg',
                    'foreground_mask': 'target_foreground.png',
                    'boxes': t_boxes,
                    'text_regions': t_text
                },
                'source': {
                    'id': spec['source_id'],
                    'title': spec['source_title'],
                    'image': 'source.jpg',
                    'foreground_mask': 'source_foreground.png',
                    'boxes': s_boxes,
                    'text_regions': s_text
                }
            }
            save_json(defn_path, meta)

            group_results.append(meta)
            contact_sheet_items.append((key, sheet_path))

    finally:
        if annotator is not None:
            annotator.close()
        db.close()

    # 5. 生成 9 组全量拼接大图 CONTACT_SHEET_ALL9.jpg (3 列 x 3 行)
    print("\n正在生成 9 组全量汇总审核大图 CONTACT_SHEET_ALL9.jpg...")
    tile_w, tile_h = 920, 580
    grid_cols, grid_rows = 3, 3
    all9_canvas = Image.new('RGB', (tile_w * grid_cols, tile_h * grid_rows + 70), '#ffffff')
    all9_draw = ImageDraw.Draw(all9_canvas)

    banner_font = load_font(24)
    all9_draw.text((30, 20), "PerLAGE 9组商品图对人工审核大图 (左侧TARGET目标商品，右侧SOURCE源商品 | 绿框:主体边缘，红框:文字区域)",
                   fill='#111827', font=banner_font)

    for i, (key, spath) in enumerate(contact_sheet_items):
        col = i % grid_cols
        row = i // grid_cols
        tile = Image.open(spath)
        all9_canvas.paste(tile, (col * tile_w, 70 + row * tile_h))

    all9_path = OUTPUT_BASE / 'CONTACT_SHEET_ALL9.jpg'
    all9_canvas.save(all9_path, quality=92)
    print(f"全量总览图已保存: {all9_path}")

    # 6. 生成 OVERVIEW.md
    print("正在编写 OVERVIEW.md 综合审查说明文档...")
    md_lines = [
        "# 9 组图对选取与人工审查包 (Review Package 20261108)",
        "",
        "依据公子指示，执行现行第二版门槛抽样方案（`public_shape_gate_uniform_v2`）及既有约定，已完成 9 组图对的选取、主体分割（SAM 2.1）、文字识别（RapidOCR）标注及核对预览图生成。",
        "",
        "## 9 组图对总览表",
        "",
        "| 序号 | 组别标识 | 中文组名 | 目标商品编号 (Target ASIN) | 源商品编号 (Source ASIN) | 外形相似度 (Shape Sim) | 选取状态 | 审核备注 |",
        "|:---:|:---|:---|:---|:---|:---:|:---|:---|",
    ]

    status_badge_map = {
        'strictly_selected_uniform_draw': '标准门槛均匀抽中',
        'frozen_historical_reuse': '约定直接复用',
        'screened_high_similarity': '受控库高相似候选'
    }

    for idx, r in enumerate(group_results, start=1):
        g_id = r['group']
        name = r['name']
        t_id = r['target']['id']
        s_id = r['source']['id']
        sim = r['shape_similarity']
        st = status_badge_map.get(r['selection_status'], r['selection_status'])
        notes = r['selection_notes']
        md_lines.append(f"| {idx} | `{g_id}` | {name} | `{t_id}` | `{s_id}` | **{sim:.4f}** | {st} | {notes} |")

    md_lines.extend([
        "",
        "---",
        "",
        "## 审核图片资源索引",
        "",
        f"- **全量 9 组总览审查图 (3x3 网格拼接大图)**: [`CONTACT_SHEET_ALL9.jpg`](file:///{all9_path.as_posix()})",
        "",
        "### 各组独立目录与单组核对图：",
    ])

    for r in group_results:
        g_id = r['group']
        name = r['name']
        g_path = (groups_dir / g_id).resolve()
        cs_path = g_path / 'contact_sheet.jpg'
        md_lines.append(f"1. **{name} (`{g_id}`)**: [浏览目录](file:///{g_path.as_posix()}) | [单组核对图](file:///{cs_path.as_posix()})")
        md_lines.append(f"   - **Target**: `{r['target']['id']}` - {r['target']['title']}")
        md_lines.append(f"   - **Source**: `{r['source']['id']}` - {r['source']['title']}")
        md_lines.append(f"   - **标注文件**: `target.jpg`, `source.jpg`, `target_foreground.png`, `source_foreground.png`, `definition.json`")

    md_lines.extend([
        "",
        "---",
        "",
        "## 选取机制与技术说明",
        "",
        "1. **校准阈值标准**: 公开开发集校准阈值为 $\\tau = 0.601081$。对于理发椅（02）、曲奇面团（03）、接发（05）、牛仔裤（06）、U型牙刷（09），在受控候选集合内均有多条候选稳定跨越门槛，并在严格排除同编号、同主图、类型冲突后，由伪随机哈希拒绝取样（`uniform_draw`）在 seed 42 下公平抽选确定。",
        "2. **洗发水组直接复用**: 第七组洗发水（07）严格遵照公子指示，直接复用多芬（Dove）泡沫洗手液作为源商品、艾凡达（Aveda）洗发水作为目标商品及其既有掩码，未重新触发选取。",
        "3. **相机、窗帘与音箱的同构筛查**: 相机（01）、窗帘（04）、音箱（08）在原短名单抽样中由于候选库稀疏或外形特殊，未在有限哈希前缀中自动命中 $\\ge 0.601081$。本次在全部模拟可控商品库中检索对应同构类型（PTZ球机、木纹窗帘、箱式音箱），筛选出外形最具代表性的真实商品供公子人工审核确认。",
        ""
    ])

    overview_file = OUTPUT_BASE / 'OVERVIEW.md'
    overview_file.write_text('\n'.join(md_lines), encoding='utf-8')
    print(f"综合审查说明文档已保存: {overview_file}")

    # 保存完整的 JSON 汇总清单
    bundle = {
        'schema_version': 2,
        'package_type': 'review_package_9groups',
        'calibrated_threshold': 0.6010810730721305,
        'created_at': '2026-11-08',
        'groups': group_results,
        'overview_markdown': str(overview_file.resolve()),
        'contact_sheet_all': str(all9_path.resolve())
    }
    save_json(OUTPUT_BASE / 'REVIEW_MANIFEST.json', bundle)
    print(f"审查清单 JSON 已保存: {OUTPUT_BASE / 'REVIEW_MANIFEST.json'}")
    print("\n========== 9 组审核包生成全部完成！ ==========")


if __name__ == '__main__':
    build_review_package()
