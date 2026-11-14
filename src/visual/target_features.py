"""只从可追溯公开商品字段提取特点；不读购买轨迹、正式请求或名次。"""
from __future__ import annotations

import json
import re
import sqlite3
from typing import Any
from src.paths import ProjectPath as Path, DATA

CATALOG_DB_PATH = DATA / 'public/product_metadata.json'
REVIEW_PATH = DATA / 'public/product_feature_review.json'
CATEGORY_DISPLAY_NAMES = {
    '01_camera': 'security camera', '02_chair': 'barber chair',
    '03_cookies': 'cookie dough', '04_curtain': 'curtains',
    '05_hair_extension': 'hair extensions', '06_jeans': 'jeans',
    '07_shampoo': 'shampoo', '08_speaker': 'bluetooth speaker',
    '09_toothbrush': 'toothbrush',
}
TARGET_IDS = ('B08V134QWR','B09CYFD5BX','B08Z8MHWVS','B086PW7TWX','B01460C2I2',
              'B09HH96NJY','B00BDJODWI','B01N47U1VI','B09J81RMLD')


def _reviews():
    return json.loads(REVIEW_PATH.read_text(encoding='utf-8')).get('products', {}) if REVIEW_PATH.is_file() else {}


# 兼容旧调用名；内容来自附有逐字段出处的公开审核文件，禁止无出处的人工表。
CURATED_PRODUCT_FEATURES = _reviews()
CURATED_TARGET_FEATURES = {k: v for k, v in CURATED_PRODUCT_FEATURES.items() if k in TARGET_IDS}
CURATED_SOURCE_FEATURES = {k: v for k, v in CURATED_PRODUCT_FEATURES.items() if k not in TARGET_IDS}

# 只识别明确的命名品牌。未命中时留空，绝不把首个品类词充作品牌。
NAMED_BRANDS = (
    'Cherish Baby Care', 'Dr. Collins', 'Big Dipper Dough Co', 'Natural Decadence',
    'Touch of ECO', 'Pepperidge Farm', 'Blue Microphones', 'Exclusive Home',
    'Herbal Essences', 'Desert Essence', 'Natural Riches', 'Irene House',
    'Acoustic Audio', 'EMMA + OLIVER', 'STEVE & ANDY\'S', 'Levi\'s',
    'L\'Oreal Paris', 'H.VERSAILTEX', 'RYB HOME', 'Nikon', 'Canon', 'Enster',
    'SUNBA', 'LTHDD', 'Buy-Rite', 'Dove', 'Aveda', 'Cinch', 'Kicker', 'S-noilite',
    'Tthxqing', 'Rockville', 'Fujifilm', 'Panasonic', 'Foscam', 'Amcrest',
    'Netvue', 'DELUXIO', 'figatia', 'ZOSI', 'INNOREL', 'ZEEPORTE', 'SSLine',
    'Giantex', 'Calico Designs', 'AWQM', 'FMOGG', 'Serta', 'X-VOLSPORT',
    'Rivet', 'Nouhaus', 'LUARANE', 'Dolonm', 'Merrick Lane', 'Funnylife',
    'Casual Home', 'Tyyps', 'Annie\'s', 'Maxine\'s Heavenly', 'LATIN SABOR',
    'RED SAKURA', 'The Cravory', 'Pamela\'s', 'Hershey\'s', 'Blacksherbet',
    'Maison Fossier', 'Glico', 'Neelam Foodland', 'Oreo', 'Kay\'s Naturals',
    'Cocosette', 'Oberlander\'s', 'Lotus', 'Ambesonne', 'Yaliaprint', 'Brawvy',
    'Valea Home', 'Aquazolax', 'Rajlinen', 'Libaoge', 'ZGCLHHM', 'Salerm',
    'Ayur', 'Kristin Ess', 'Chagrin Valley', 'Truss', 'OGX', 'GIOVANNI',
    'Authentic Beauty Concept', 'Luseta', 'SheaMoisture', 'KUNDAL', 'ceceMED',
    'Infuse My. Colour', 'Mera', 'Puracy', 'JBL', 'Tgoon', 'FFENFEI', 'Alpine',
    'ECOXGEAR', 'Totola', 'Lanzar', 'GoHawk', 'Niaviben', 'Pantone', 'LESHP',
    'Fmystery', 'LVOD', 'co2CREA', 'Jabra', 'DOITOOL', 'Simayixx', 'Alvinlite',
    'OBrush', 'TeK', 'Healifty', 'Bine-Deam55',
)


def extract_brand_from_text(title, bullet_points=None, description=None):
    for text in [*(bullet_points or []), description or '']:
        m = re.search(r'\b(?:Brand|Manufacturer|Design House)\s*:\s*([^\n;,]{2,45})', str(text), re.I)
        if m:
            return m.group(1).strip().rstrip('.')
    m = re.search(r'\bby\s+([A-Za-z][A-Za-z0-9 .&\'-]{1,35}?)(?=\s+-|[,;]|$)', title, re.I)
    if m:
        return m.group(1).strip()
    for brand in sorted(NAMED_BRANDS, key=len, reverse=True):
        if re.search(r'(?<!\w)' + re.escape(brand) + r'(?!\w)', title, re.I):
            return brand
    return ''


def _clean_attribute_candidate(text):
    value = str(text).strip().strip('-.,;"\'()[]')
    if not 3 <= len(value) <= 160 or re.fullmatch(r'[\d.\s"\'xX*\-]+', value):
        return None
    if value.casefold() in {'design house', 'dummy_attr', 'high quality', 'standard item'}:
        return None
    if any(w in value.casefold() for w in ('free shipping', 'money back', 'warranty',
                                           'return policy', 'action:', 'click[', 'think[')):
        return None
    return value


def build_natural_buyer_intent(category, attributes, title=None):
    attrs = [v.lower() for a in attributes if (v := _clean_attribute_candidate(a))]
    if not attrs:
        raise ValueError('没有有依据的特点，不能用虚构通用卖点补齐')
    cat = str(category).lower().strip()
    if not cat:
        raise ValueError('缺少公开商品品类')
    plural = (cat.endswith('s') and not cat.endswith(('ss', 'us'))) or cat in ('cookie dough', 'equipment', 'shampoo', 'hand wash')
    noun = cat if plural else ('an ' if cat[0] in 'aeiou' else 'a ') + cat
    pairs = [(attrs[0], attrs[min(1,len(attrs)-1)]),
             (attrs[0], attrs[min(2,len(attrs)-1)]),
             (attrs[min(1,len(attrs)-1)], attrs[-1])]
    features = [a if a == b else a + ' and ' + b for a,b in pairs]
    queries = [f'I want to buy {noun} with {features[0]}.',
               f'I would like to purchase {noun} featuring {features[1]}.',
               f'Looking to buy {noun} with {features[2]}.']
    return queries[0], queries


def _public_product(asin, path):
    if not path.is_file():
        raise FileNotFoundError(f'缺少本项目公开商品元数据: {path}')
    if path.suffix.lower() == '.json':
        item = json.loads(path.read_text(encoding='utf-8')).get('products', {}).get(asin)
        if item is None:
            raise ValueError(f'公开元数据中没有商品 {asin}，不能虚构特点')
        return item
    with sqlite3.connect(f'file:{path.as_posix()}?mode=ro', uri=True) as conn:
        row = conn.execute('SELECT data, category, query FROM products WHERE asin=?', (asin,)).fetchone()
        if row is None:
            raise ValueError(f'公开元数据中没有商品 {asin}，不能虚构特点')
        product = json.loads(row[0])
        product.setdefault('category', row[1]); product.setdefault('query', row[2])
        return product


def extract_product_features(asin, db_path=None):
    product = _public_product(asin, Path(db_path or CATALOG_DB_PATH))
    title = str(product.get('Title') or product.get('name') or '')
    # 显式数据源用于新选图；已审核特点只对相同的公开商品正文生效。
    reviewed = _reviews().get(asin) if db_path is None else None
    if reviewed is not None:
        if reviewed['title'] != title:
            raise ValueError(f'审核特点与当前公开标题不一致: {asin}')
        for evidence in reviewed['feature_evidence']:
            field = product.get(evidence['field'], '')
            field = field[evidence['index']] if 'index' in evidence else field
            if evidence['quote'] not in str(field):
                raise ValueError(f'审核特点缺少原文证据: {asin}/{evidence["feature"]}')
        result = dict(reviewed)
        result['core_attributes'] = list(reviewed['attributes'])
        result['attributes'] = list(reviewed['attributes'])
        result['feature_evidence'] = list(reviewed['feature_evidence'])
        # 精炼训练正文与完整记忆属性分开保存，不能丢掉已有的真实共有特点。
        for index, raw in enumerate(product.get('Attributes') or []):
            value = _clean_attribute_candidate(raw)
            if value and value.casefold() not in {a.casefold() for a in result['attributes']}:
                result['attributes'].append(value)
                result['feature_evidence'].append(dict(feature=value, field='Attributes', index=index, quote=raw))
        return result
    bullets = product.get('BulletPoints') or product.get('small_description') or []
    description = product.get('Description') or product.get('full_description') or ''
    explicit = product.get('Brand') or product.get('brand')
    brand = str(explicit).strip() if explicit else extract_brand_from_text(title, bullets, description)
    attributes, evidence = [], []
    for index, raw in enumerate(product.get('Attributes') or []):
        value = _clean_attribute_candidate(raw)
        if value and value.casefold() not in {a.casefold() for a in attributes}:
            attributes.append(value)
            evidence.append(dict(feature=value, field='Attributes', index=index, quote=raw))
    if not attributes:
        for index, bullet in enumerate(bullets):
            for fragment in re.split(r'(?<!\d)\.(?!\d)|[;\n]', str(bullet)):
                fragment = fragment.split(':', 1)[-1].strip()
                value = _clean_attribute_candidate(fragment)
                if value and value not in attributes:
                    attributes.append(value)
                    evidence.append(dict(feature=value, field='BulletPoints', index=index, quote=fragment))
                if len(attributes) == 3:
                    break
            if len(attributes) == 3:
                break
    if not attributes:
        # 稀疏商品保留实际标题短语，明确它是标题摘录，不生成不存在的属性。
        for fragment in re.split(r'[,|;]|\s+-\s+', title):
            if len(fragment) > 160:
                fragment = fragment[:150].rsplit(' ', 1)[0]
            value = _clean_attribute_candidate(fragment)
            if value and value not in attributes:
                attributes.append(value)
                evidence.append(dict(feature=value, field='Title', quote=fragment))
            if len(attributes) == 3:
                break
    if not attributes:
        raise ValueError(f'商品 {asin} 缺少有依据的特点，须完善公开元数据')
    category = str(product.get('query') or product.get('category') or 'product')
    canonical, queries = build_natural_buyer_intent(category, attributes, title)
    return dict(asin=asin, title=title, brand=brand, category=category, attributes=attributes,
                canonical_sentence=canonical, training_queries=queries,
                brand_evidence=dict(verified=bool(brand), reason='explicit_catalog_or_named_title_brand' if brand else 'unknown_public_brand'),
                feature_evidence=evidence, source='local_public_catalog_fields')


def format_retrieval_text(title: str, brand: str, features: Any) -> str:
    lines = [f'Title: {title.strip()}']
    if brand and str(brand).strip():
        lines.append(f'Brand: {str(brand).strip()}')
    values = [str(v).strip() for v in features if str(v).strip()] if isinstance(features, (list,tuple)) else [str(features).strip()] if features else []
    if values:
        lines.append('Features: ' + ', '.join(values))
    return '\n'.join(lines)
