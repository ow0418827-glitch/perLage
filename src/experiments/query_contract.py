"""显式纯文字或独立图文的正式查询约定；两种模式均不得静默回退。"""
import hashlib
import json
from src.paths import ProjectPath as Path
from src.visual.audit import read_json
from src.visual.image_utils import sha256_file




class QueryImagePending(RuntimeError):
    """未提供独立生成并经核对的正式查询图片。"""


def pending_query_spec(target, query_mode='joint'):
    if query_mode == 'text':
        return dict(schema_version=2, status='awaiting_evaluation_manifest', query_mode='text',
                    target_public_id=target['id'], query_image_required=False,
                    formal_instructions_read_during_generation=False,
                    requirements=['评估时读取已冻结的二十条纯文字购买意向', '本模式不生成或读取查询图片'])
    if query_mode != 'joint':
        raise ValueError('正式查询模式只能是纯文字或图文联合')
    return dict(schema_version=1, status='pending_generation', target_public_id=target['id'],
                reference_image=target['image'], reference_sha256=target['image_sha256'],
                instruction_policy='vague_similar_pictured_product',
                requirements=['单独生成参考图，不复用商品原图', '不能出现明确的商品名称',
                              '不能出现可识别的品牌名称或品牌标志', '保留大致商品形态与用途线索'],
                query_image_generation_implemented=False,
                expected_manifest=dict(status='ready', target_public_id=target['id'], image='待填写独立查询图路径', sha256='待填写图片内容摘要',
                    review=dict(no_explicit_product_name=True, no_brand=True, reviewer='待填写实际核对者')))


def load_formal_queries(manifest_path, target, *, query_mode=None):
    from src.experiments.formal_queries import VAGUE_INSTRUCTIONS
    path = Path(manifest_path)
    if not path.is_file():
        raise QueryImagePending('独立模糊查询图尚未生成；请按运行目录中的查询图片要求补齐后再评估')
    manifest = read_json(path)
    if manifest.get('target_public_id') != target['id']:
        raise ValueError('正式查询未绑定当前公开目标商品')
    mode = manifest.get('query_mode', 'joint')
    if mode not in ('text', 'joint'):
        raise ValueError('正式查询模式只能是纯文字或图文联合')
    if query_mode is not None and query_mode != mode:
        raise ValueError('运行指定的正式查询模式与冻结清单不一致')
    instructions = manifest.get('instructions')
    if instructions is not None:
        if (not isinstance(instructions, list) or len(instructions) != 20
                or any(not isinstance(text, str) or not text.strip() for text in instructions)
                or len({' '.join(text.split()).casefold() for text in instructions}) != 20):
            raise ValueError('正式询问必须恰好包含二十条非空且不重复的固定文字')
        expected = manifest.get('instructions_sha256')
        actual_text = hashlib.sha256(json.dumps(instructions, ensure_ascii=False, separators=(',', ':')).encode('utf-8')).hexdigest()
        if expected is not None and actual_text != expected:
            raise ValueError('正式询问正文摘要不符')
    elif mode == 'text':
        raise ValueError('纯文字模式必须显式提供二十条正式询问，不能回退到默认泛指请求')
    else:
        instructions = VAGUE_INSTRUCTIONS
    if mode == 'text':
        if manifest.get('status') != 'ready':
            raise ValueError('纯文字正式询问清单尚未就绪')
        if manifest.get('image') is not None or manifest.get('sha256') is not None:
            raise ValueError('纯文字查询模式不能附带参考图或图片摘要')
        return [dict(task_id=f'feature_{i:02d}', instruction=text, query_mode='text',
                     query_image=None, query_image_sha256=None)
                for i, text in enumerate(instructions, 1)]
    review = manifest.get('review', {})
    if (manifest.get('status') != 'ready' or review.get('no_explicit_product_name') is not True
            or review.get('no_brand') is not True or not str(review.get('reviewer', '')).strip()):
        raise QueryImagePending('正式查询图需要单独生成，并核对没有明确商品名称或品牌')
    if not isinstance(manifest.get('image'), str) or not manifest['image'].strip():
        raise QueryImagePending('正式查询图片路径尚未提供')
    image = Path(manifest['image'])
    image = (image if image.is_absolute() else path.resolve().parent / image).resolve()
    if not image.is_file():
        raise QueryImagePending('正式查询图片不存在')
    actual = sha256_file(image)
    if actual != manifest.get('sha256') or actual == target.get('image_sha256'):
        raise ValueError('查询图摘要不符，或误用了目标商品原图')
    prefix = manifest.get('task_id_prefix', 'vague')
    if prefix not in ('feature', 'vague'):
        raise ValueError('正式询问任务编号前缀无效')
    if manifest.get('query_image_weight', 0.5) != 0.5 or manifest.get('query_text_weight', 0.5) != 0.5:
        raise ValueError('正式图文查询必须采用图文各半的固定融合权重')
    return [dict(task_id=f'{prefix}_{i:02d}', instruction=text, query_mode='joint', query_image=str(image),
                 query_image_sha256=actual) for i, text in enumerate(instructions, 1)]


def query_retriever_options(queries):
    modes = {q.get('query_mode', 'joint') for q in queries}
    if not queries or len(modes) != 1 or not modes <= {'text', 'joint'}:
        raise ValueError('同一评估须使用明确一致的正式查询模式')
    mode = next(iter(modes))
    return dict(query_mode=mode, query_image_weight=0.0 if mode == 'text' else 0.5)


def verify_query_input(query):
    """复算只校验实际模式所需输入；纯文字模式永远不读取图片。"""
    if not isinstance(query.get('instruction'), str) or not query['instruction'].strip():
        raise ValueError('正式询问正文无效')
    mode = query.get('query_mode', 'joint')
    if mode == 'text':
        if query.get('query_image') is not None or query.get('query_image_sha256') is not None:
            raise ValueError('纯文字正式询问不能附带图片')
    elif mode == 'joint':
        if not query.get('query_image') or not query.get('query_image_sha256'):
            raise QueryImagePending('图文正式询问缺少独立参考图或摘要')
        if not Path(query['query_image']).is_file():
            raise QueryImagePending('正式查询图片不存在')
        if sha256_file(query['query_image']) != query['query_image_sha256']:
            raise ValueError('正式查询图片改变')
    else:
        raise ValueError('正式查询模式无效')
