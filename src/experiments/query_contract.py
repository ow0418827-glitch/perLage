"""模糊文字与独立模糊图的正式查询约定；图片生成有意保留待实现。"""
from src.paths import ProjectPath as Path
from src.visual.audit import read_json
from src.visual.image_utils import sha256_file




class QueryImagePending(RuntimeError):
    """未提供独立生成并经核对的正式查询图片。"""


def pending_query_spec(target):
    return dict(schema_version=1, status='pending_generation', target_public_id=target['id'],
                reference_image=target['image'], reference_sha256=target['image_sha256'],
                instruction_policy='vague_similar_pictured_product',
                requirements=['单独生成参考图，不复用商品原图', '不能出现明确的商品名称',
                              '不能出现可识别的品牌名称或品牌标志', '保留大致商品形态与用途线索'],
                query_image_generation_implemented=False,
                expected_manifest=dict(status='ready', target_public_id=target['id'], image='待填写独立查询图路径', sha256='待填写图片内容摘要',
                    review=dict(no_explicit_product_name=True, no_brand=True, reviewer='待填写实际核对者')))


def load_formal_queries(manifest_path, target):
    from src.experiments.formal_queries import VAGUE_INSTRUCTIONS
    path = Path(manifest_path)
    if not path.is_file():
        raise QueryImagePending('独立模糊查询图尚未生成；请按运行目录中的查询图片要求补齐后再评估')
    manifest = read_json(path)
    review = manifest.get('review', {})
    if (manifest.get('status') != 'ready' or review.get('no_explicit_product_name') is not True
            or review.get('no_brand') is not True or not str(review.get('reviewer', '')).strip()):
        raise QueryImagePending('正式查询图需要单独生成，并核对没有明确商品名称或品牌')
    image = Path(manifest.get('image', ''))
    image = (image if image.is_absolute() else path.resolve().parent / image).resolve()
    if not image.is_file():
        raise QueryImagePending('正式查询图片不存在')
    actual = sha256_file(image)
    if actual != manifest.get('sha256') or actual == target['image_sha256']:
        raise ValueError('查询图摘要不符，或误用了目标商品原图')
    if manifest.get('target_public_id') != target['id']:
        raise ValueError('正式查询图未绑定当前公开目标商品')
    return [dict(task_id=f'vague_{i:02d}', instruction=text, query_image=str(image),
                 query_image_sha256=actual) for i, text in enumerate(VAGUE_INSTRUCTIONS, 1)]
