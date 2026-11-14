from pathlib import Path
root=Path(__file__).resolve().parents[1]
p=root/'src/agent/shopping.py';text=p.read_text(encoding='utf-8')
start=text.index("''' Setting up webshop environment'''")
end=text.index("initial_prompt = INITIAL_PROMPTS",start)
segment=text[start:end]
prefix='''"""商城页面解析与动作环境客户端；模型推理留在智能体策略中。"""
import copy
import json
import re
from src.paths import ProjectPath as Path
from src.agent.memory.multimodal_memory import (
    extract_page_images, save_memory_image, snapshot_product_image)
'''
segment=segment.replace("''' Setting up webshop environment'''",'')
segment=segment.replace('WEBSHOP_URL = f"http://localhost:3000/"','DEFAULT_WEBSHOP_URL = "http://localhost:3000/"')
segment=segment.replace('http_timeout=(5, 30), **kwargs):','http_timeout=(5, 30), base_url=DEFAULT_WEBSHOP_URL, **kwargs):')
segment=segment.replace("base = WEBSHOP_URL.rstrip('/')","base = base_url.rstrip('/')")
segment=segment.replace('image_cache_root=None, image_timeout=(10, 30), image_max_attempts=2):',
                        'image_cache_root=None, image_timeout=(10, 30), image_max_attempts=2, base_url=DEFAULT_WEBSHOP_URL):')
segment=segment.replace('self.sessions = {}','self.sessions = {}\n    self.base_url = base_url')
segment=segment.replace('http_timeout=self.http_timeout)','http_timeout=self.http_timeout, base_url=self.base_url)')
(root/'src/webshop/client.py').write_text(prefix+segment,encoding='utf-8')
text=text[:start]+'from src.webshop.client import webshopEnv\n\n\n'+text[end:]
p.write_text(text,encoding='utf-8')
p=root/'src/experiments/shared_shopping.py';text=p.read_text(encoding='utf-8')
text=text.replace("    shopping.WEBSHOP_URL=f'http://127.0.0.1:{PORT}/'\n",'')
text=text.replace("process=None;shopping.WEBSHOP_URL=f'http://127.0.0.1:{PORT}/'","process=None")
text=text.replace('shopping.webshopEnv(instruction_overrides=',"shopping.webshopEnv(base_url=f'http://127.0.0.1:{PORT}/',instruction_overrides=")
p.write_text(text,encoding='utf-8')
p=root/'src/experiments/recall_depth_evaluation.py';text=p.read_text(encoding='utf-8')
text=text.replace("process=service.start_server(root);agent.WEBSHOP_URL=f'http://127.0.0.1:{service.PORT}/'","process=service.start_server(root)")
text=text.replace('agent.webshopEnv(instruction_overrides=',"agent.webshopEnv(base_url=f'http://127.0.0.1:{service.PORT}/',instruction_overrides=")
p.write_text(text,encoding='utf-8')
print('Environment client extracted; base URL is an explicit instance setting.')
