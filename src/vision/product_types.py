"""只从公开名称识别粗类型；未知或多义名称需要逐商品公开核对。"""
import re
from src.vision.audit import digest

TYPE_POLICY='public_title_coarse_conflicts_v2'
RULES={
    'seating':r'\b(?:chairs?|stools?|armchairs?|ottomans?|recliners?|seat cushions?)\b|椅|凳|坐垫',
    'oral_care':r'\b(?:toothbrush(?:es)?|toothpaste|teethers?|dental|floss)\b|牙刷|牙膏|口腔|牙胶',
    'personal_care':r'\b(?:shampoo|conditioner|body wash|lotion|hair oil|hair treatment|shower gel)\b|洗发|护发|沐浴|乳液',
    'hair_extensions':r'\b(?:wigs?|hair(?:\s+\w+){0,2}\s+extensions?|wefts?|lace frontal|hair bundles?)\b|假发|接发',
    'camera':r'\b(?:cameras?|camcorders?|camera lenses|photography)\b|相机|摄影|摄像',
    'audio':r'\b(?:speakers?|soundbars?|loudspeakers?|subwoofers?|headphones?)\b|音箱|音响|耳机',
    'apparel':r'\b(?:jeans|pants|trousers|dresses|shirts?|jackets?|leggings)\b|牛仔裤|长裤|上衣|裙|服装',
    'curtains':r'\b(?:curtains?|drapes?|window valance)\b|窗帘|帷幔',
    'tableware':r'\b(?:mugs?|cups?|bowls?|plates?|cutlery)\b|餐具|杯|碗|盘',
}
# 粗类型间默认排除明显不同的用途；两种护发用途保留宽松兼容。
COMPATIBLE={frozenset(('personal_care','hair_extensions'))}


def classify_public_type(product):
    title=product['title']
    matches={key:re.findall(pattern,title,re.I) for key,pattern in RULES.items()}
    matches={key:value for key,value in matches.items() if value}
    if re.search(r'\bair[ -]condition',title,re.I):matches.pop('personal_care',None)
    review=product.get('type_review')
    if review is not None:
        if (not isinstance(review,dict) or set(review)-{'category','reviewer','basis','title_sha256'}
                or review.get('category') not in RULES or not str(review.get('reviewer','')).strip()
                or not str(review.get('basis','')).strip() or review.get('title_sha256')!=digest(title)):
            raise ValueError('公开类型核对须绑定当前名称并记录核对者、依据和有效粗类型')
        return dict(category=review['category'],method='public_review',review=review,rules_sha256=TYPE_RULES_SHA256)
    if len(matches)!=1:
        return dict(category=None,method='unresolved',matches=matches,rules_sha256=TYPE_RULES_SHA256)
    category=next(iter(matches))
    return dict(category=category,method='public_title_rule',matches=matches[category],rules_sha256=TYPE_RULES_SHA256)


def type_compatibility(source,target):
    a,b=classify_public_type(source),classify_public_type(target)
    resolved=a['category'] is not None and b['category'] is not None
    compatible=resolved and (a['category']==b['category'] or frozenset((a['category'],b['category'])) in COMPATIBLE)
    return dict(compatible=bool(compatible),resolved=resolved,source=a,target=b,policy=TYPE_POLICY)


from pathlib import Path
import hashlib
TYPE_RULES_SHA256=digest(dict(policy=TYPE_POLICY,rules=RULES,
    compatible=[sorted(x) for x in sorted(COMPATIBLE,key=lambda x:sorted(x))],
    implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
