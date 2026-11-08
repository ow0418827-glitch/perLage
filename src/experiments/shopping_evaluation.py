"""可选的下游购物评估适配；默认停止，复用同一共享池和模糊查询。"""
from src.paths import ProjectPath as Path
import numpy as np

from src.visual.audit import read_json, save_json
from src.visual.image_utils import sha256_file


def run_shopping_evaluation(method_root, *, model_config, allow_shopping=False,
                            num_steps=15, environment_factory=None, device='cuda',
                            prompt_path=None, prompt_profile=None):
    method_root = Path(method_root).resolve()
    stop = method_root.parents[1] / 'STOP_SHOPPING_REQUESTED.json'
    if not allow_shopping or not stop.is_file() or read_json(stop).get('shopping_allowed') is not True:
        raise PermissionError('购物默认停止；未来正式测试须显式启用，并先更新本次运行的停止购买标记')
    from src.agent.config.llm_config import load_model_config
    from src.agent.shopping import webshopEnv, webshop_run_rap
    from src.agent.prompts.webshop_prompt import INITIAL_PROMPTS
    from src.agent.prompts.loader import load_shopping_prompt
    from src.visual.visual_retrieval import VisualMemoryRetriever
    from src.experiments.shared_evaluation import make_eval_encoder
    plan = read_json(method_root / 'evaluation/evaluation_plan.json')
    from src.experiments.pipeline import load_frozen
    frozen = load_frozen(method_root)
    if plan['frozen_image_sha256'] != frozen['image_sha256']:
        raise ValueError('下游评估没有绑定当前冻结图片')
    if plan.get('scope') != 'shared' or sha256_file(plan['pool_path']) != plan['pool_sha256']:
        raise ValueError('下游测试须复用同一个完整冻结共享池')
    verification = read_json(method_root / 'evaluation/independent_verification.json')
    if (verification['status'] != 'passed'
            or verification['evaluated_file_sha256'] != sha256_file(method_root / 'evaluation/evaluation.json')):
        raise ValueError('下游评估需要先通过真实检索复算')
    rows = read_json(method_root / 'evaluation/evaluation.json')
    settings = load_model_config(model_config)
    client = settings.client()
    guidance = load_shopping_prompt(prompt_path, profile=prompt_profile)
    from src.visual.audit import freeze_json
    freeze_json(method_root / 'shopping/PROMPT.json', guidance)
    encoder = make_eval_encoder(device, plan.get('encoder_config'))
    outcomes = []
    try:
        for row in rows:
            bank = read_json(row['bank_path'])
            if len(bank) != plan['pool_count'] or sha256_file(row['bank_path']) != row['bank_sha256']:
                raise ValueError('下游测试的共享池副本改变')
            if any(e['RecordKind'] != 'purchase_memory' for e in bank):
                raise ValueError('模拟图库不具备购买轨迹，不作为下游购物示例')
            for query in plan['queries']:
                session = f"perlage_{row['bank']}_{row['condition']['label']}_{query['task_id']}"
                env = (environment_factory(query, session) if environment_factory else
                       webshopEnv(instruction_overrides={session: query['instruction']}))
                ranker = VisualMemoryRetriever(encoder, row['bank_path'], cache_dir=None)
                reward, record, _ = webshop_run_rap(
                    session, '', '', INITIAL_PROMPTS['PROMPT1'], env, client,
                    settings.model, settings.provider, None, bank,
                    {'Instruction': np.zeros((len(bank), 1))}, num_steps,
                    benign=True, test=True, query_image=query['query_image'],
                    visual_retriever=ranker, retrieval_candidates=3, retrieval_top_k=3,
                    vision_input=True, honest_reward=True, memory_mode='full',
                    memory_image_input=True, memory_path=row['bank_path'],
                    policy_memory_image_field='RetrievalImage',
                    auxiliary_prompt=guidance['text'],
                    original_instruction=query['instruction'])
                outcomes.append(dict(condition=row['condition'], bank=row['bank'], task_id=query['task_id'],
                                     environment_reward=reward, record=record))
                save_json(method_root / 'shopping/outcomes.json', outcomes)
    finally:
        encoder.close()
    return outcomes
