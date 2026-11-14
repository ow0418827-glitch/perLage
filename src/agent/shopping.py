"""Run WebShop RAP experiments."""
from src.paths import relocate_io_path

import os
import sys
import yaml
import json
import hashlib
import base64
import mimetypes
from src.paths import ProjectPath as Path
import numpy as np
import torch
import argparse
import time
import random
import re
import copy
import logging
import requests
from datetime import datetime
from openai import OpenAI
# text embedding model
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from src.agent.memory.multimodal_memory import (
    extract_page_images, save_memory_image, validated_image_bytes,
    memory_image_inputs, persist_local_image, snapshot_product_image,
)
from src.agent.memory.memory_pairing import configure_text_args, validate_paired_memory
from src.agent.prompts.webshop_prompt import INITIAL_PROMPTS


from src.paths import CONFIGS, DATA, OUTPUT

with open(relocate_io_path(CONFIGS / 'base_config.yaml'), encoding="utf-8") as reader:
    config = yaml.safe_load(reader)

def setup_logger():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = str(OUTPUT / "log")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"program_output_{timestamp}.log")
    log_file = open(relocate_io_path(log_path), "w", encoding="utf-8")
    return log_file

# Initialize LLM
def get_api_key(provider: str, api_key_env: str | None, api_key_file: str | None):
    """Read a credential without persisting it in source code or experiment output."""
    default_env = "MINIMAX_API_KEY" if provider == "minimax" else "OPENAI_API_KEY"
    api_key = os.getenv(api_key_env or default_env, "").strip()

    if not api_key and api_key_file:
        with open(relocate_io_path(api_key_file), encoding="utf-8") as key_file:
            api_key = key_file.read().strip()
    # Preserve the original RAP workflow for OpenAI users who keep this local file.
    if not api_key and provider == "openai" and os.path.exists("OpenAI_api_key.txt"):
        with open(relocate_io_path("OpenAI_api_key.txt"), encoding="utf-8") as key_file:
            api_key = key_file.read().strip()
    if not api_key:
        raise ValueError(
            f"No API key found. Set {api_key_env or default_env}, or pass --api_key_file."
        )
    return api_key


def init_llm(provider: str, api_key_env: str | None, api_key_file: str | None, api_base: str | None):
    """Create an OpenAI-compatible client for OpenAI or MiniMax."""
    api_key = get_api_key(provider, api_key_env, api_key_file)
    if provider == "minimax":
        api_base = api_base or "https://api.minimaxi.com/v1"
    return OpenAI(api_key=api_key, base_url=api_base)


def parse_action(text):
    """Accept one complete action; never execute ordinary explanatory prose."""
    value = text.strip()
    if value.lower().startswith('action:'):
        value = value.split(':', 1)[1].strip()
    if not re.fullmatch(r'(search|click|think)\[[^\r\n]+\]', value):
        return '__invalid_action_format__'
    return value


class PolicyActionError(RuntimeError):
    """A bounded policy call failed without an executable action."""
    recoverable = True

    def __init__(self, message, trace=None):
        super().__init__(message)
        self.trace = copy.deepcopy(trace or {})


def _response_field(value, name, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _reasoning_json_value(value):
    """Preserve returned reasoning fields as JSON data, including SDK models."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _reasoning_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_reasoning_json_value(item) for item in value]
    if hasattr(value, 'model_dump'):
        try:
            dumped = value.model_dump(mode='json')
        except TypeError:
            dumped = value.model_dump()
        return _reasoning_json_value(dumped)
    if hasattr(value, 'to_dict'):
        return _reasoning_json_value(value.to_dict())
    if hasattr(value, '__dict__'):
        return {key: _reasoning_json_value(item) for key, item in vars(value).items()
                if not key.startswith('_')}
    return str(value)


def _policy_response_metadata(completion, attempt, token_budget, *, save_reasoning=True):
    """Audit response status and retain provider-returned reasoning by default."""
    choices = _response_field(completion, 'choices', []) or []
    choice = choices[0] if choices else None
    message = _response_field(choice, 'message')
    text = _response_field(message, 'content')
    reasoning_content = _response_field(message, 'reasoning_content')
    reasoning_details = _response_field(message, 'reasoning_details')
    reasoning_present = bool(reasoning_content or reasoning_details)
    usage = _response_field(completion, 'usage')
    token_counts = {}
    for source, label in (('prompt_tokens', 'PromptTokens'), ('completion_tokens', 'CompletionTokens'),
                          ('total_tokens', 'TotalTokens')):
        value = _response_field(usage, source)
        if isinstance(value, int) and not isinstance(value, bool):
            token_counts[label] = value
    details = _response_field(usage, 'completion_tokens_details')
    reasoning_tokens = _response_field(details, 'reasoning_tokens')
    if isinstance(reasoning_tokens, int) and not isinstance(reasoning_tokens, bool):
        token_counts['ReasoningTokens'] = reasoning_tokens
    metadata = {'Attempt': attempt, 'MaxTokens': token_budget,
                'FinishReason': _response_field(choice, 'finish_reason'),
                'ContentCharacters': len(text) if isinstance(text, str) else 0,
                'ReasoningPresent': reasoning_present, 'Usage': token_counts,
                'ReasoningSaved': bool(save_reasoning and reasoning_present)}
    if save_reasoning:
        if reasoning_content is not None:
            metadata['ReasoningContent'] = _reasoning_json_value(reasoning_content)
        if reasoning_details is not None:
            metadata['ReasoningDetails'] = _reasoning_json_value(reasoning_details)
    return text, metadata


def policy_retry_delay(error, failed_attempt):
  """瞬时连接、限流与服务错误退避；不增加调用次数或改变模型参数。"""
  transient = type(error).__name__ in {
      'APIConnectionError', 'APITimeoutError', 'RateLimitError',
      'ConnectionError', 'ConnectTimeout', 'ReadTimeout', 'Timeout'}
  status = getattr(error, 'status_code', None)
  transient = transient or status == 429 or (isinstance(status, int) and 500 <= status <= 599)
  return min(15 * (2 ** max(0, failed_attempt - 1)), 60) if transient else 1


def llm(prompt, model_name, client, provider, stop=["\n"], query_image=None, trace=None,
        memory_images=None, max_tokens=None, max_retries=30, save_reasoning=True):
  """Call the configured LLM and return the trimmed response."""
  retries = 0
  if max_retries < 1 or (max_tokens is not None and max_tokens < 1):
    raise ValueError('Policy token and retry limits must be positive')
  initial_token_budget = max_tokens or (512 if provider == 'minimax' else 100)
  token_budget = initial_token_budget
  retry_token_limit = max(initial_token_budget, 8192)
  call_trace = trace if trace is not None else {}
  call_trace['ResponseAttempts'] = []
  call_trace['ReasoningCaptureEnabled'] = bool(save_reasoning)
  content = prompt
  input_images = []
  content_audit = [{"type": "text", "text": prompt}]
  if query_image:
    image_path = Path(query_image)
    if memory_images is not None:
      image_path, body, metadata = validated_image_bytes(
          {'LocalPath': str(image_path.resolve())}, require_digest=False)
      mime = metadata['MimeType']
    else:
      body = image_path.read_bytes()
      mime = mimetypes.guess_type(image_path.name)[0] or 'image/png'
      metadata = {'SHA256': hashlib.sha256(body).hexdigest(), 'Bytes': len(body), 'MimeType': mime}
    encoded = base64.b64encode(body).decode('ascii')
    content = [{"type": "text", "text": prompt},
               {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}]
    query_asset = {'Role': 'query', 'LocalPath': str(image_path.resolve()), **metadata}
    input_images.append(query_asset)
    content_audit.append({'type': 'image_reference', **query_asset})
    if memory_images is not None:
      query_label = 'The preceding image is the reference image for the current user purchase request.'
      content.append({'type': 'text', 'text': query_label})
      content_audit.append({'type': 'text', 'text': query_label})
  if memory_images:
    if isinstance(content, str):
      content = [{"type": "text", "text": prompt}]
    for asset in memory_images:
      path, body, metadata = validated_image_bytes(asset)
      label = (f"Image for recalled memory Id={asset['Id']}. "
               "Associate this image with that memory's text example above.")
      encoded = base64.b64encode(body).decode('ascii')
      content.append({'type': 'text', 'text': label})
      content.append({'type': 'image_url', 'image_url': {
          'url': f"data:{metadata['MimeType']};base64,{encoded}"}})
      audited = dict(asset, LocalPath=str(path), **metadata)
      input_images.append(audited)
      content_audit.extend([{'type': 'text', 'text': label}, {'type': 'image_reference', **audited}])
  if trace is not None:
      # Exact text blocks and byte digests audit the real payload without logging base64.
      trace.update(Prompt=prompt, InputContent=content_audit, InputImages=input_images,
                   MemoryImages=[asset for asset in input_images if asset.get('Role') == 'recalled_memory'])
  while retries < max_retries:
    response_metadata = {'Attempt': retries + 1, 'MaxTokens': token_budget,
                         'FinishReason': None, 'ContentCharacters': 0,
                         'ReasoningPresent': False, 'Usage': {}, 'ReasoningSaved': False}
    try:
      request = dict(
          model=model_name,
          messages=[
              {"role": "system", "content": "You are a helpful assistant for household task."},
              {"role": "user", "content": content},
          ],
          temperature=0.5,
          # Thinking tokens share this budget. Increase it only after an empty
          # response explicitly reports that generation reached its length cap.
          max_tokens=token_budget,
          top_p=1,
          frequency_penalty=0.0,
          presence_penalty=0.0,
      )
      # MiniMax's OpenAI-compatible endpoint otherwise includes reasoning text in
      # content for reasoning-capable models, which would corrupt an action-only
      # ReAct response.  Keep that reasoning in the provider-specific field.
      if provider == "minimax":
          request["extra_body"] = {"reasoning_split": True}
      call_trace['CallAttempts'] = retries + 1
      call_trace['PolicyRequest'] = {
              'Model': model_name, 'Provider': provider, 'Temperature': request['temperature'],
              'MaxTokens': request['max_tokens'], 'MaxRetries': max_retries,
              'InitialMaxTokens': initial_token_budget, 'EmptyResponseMaxTokens': retry_token_limit,
              'SystemPrompt': request['messages'][0]['content'],
          }
      completion = client.chat.completions.create(**request)
      text, response_metadata = _policy_response_metadata(
          completion, retries + 1, token_budget, save_reasoning=save_reasoning)
      call_trace['FinishReason'] = response_metadata['FinishReason']
      if not isinstance(text, str) or not text.strip():
          raise PolicyActionError('The model returned no action content')
      call_trace['ResponseAttempts'].append(response_metadata)
      break
    except Exception as e:
      response_metadata['ErrorType'] = type(e).__name__
      call_trace['ResponseAttempts'].append(response_metadata)
      call_trace['CallErrorType'] = type(e).__name__
      retries += 1
      if retries == max_retries:
        print("Max retries reached. Execution failed")
        if isinstance(e, PolicyActionError):
          e.trace = copy.deepcopy(call_trace)
        raise e
      if isinstance(e, PolicyActionError) and response_metadata['FinishReason'] == 'length':
        token_budget = min(token_budget * 2, retry_token_limit)
      delay = policy_retry_delay(e, retries)
      response_metadata['RetryDelaySeconds'] = delay
      time.sleep(delay)

  action = parse_action(text)
  if trace is not None:
      trace.update(RawResponse=text, ParsedAction=action,
                   FormatValid=action != '__invalid_action_format__',
                   FinishReason=response_metadata['FinishReason'])
  return action




from src.webshop.client import webshopEnv


initial_prompt = INITIAL_PROMPTS[config['params'].get('initial_prompt', 'PROMPT1')]

def generate_embeddings(memory, model_embedding, *, multimodal=False):
    """Generate embeddings from memory entries for retrieval."""
    memory = [m for m in memory if m['Reward'] > 0.0]
    if config['params'].get('success', False):
      memory = [m for m in memory if m['Success']]
    print('num_retrieval',len(memory))
    embeddings = {}
    if not memory:
        return memory, embeddings
    if multimodal:
        # 图文检索的键由图片与商品名称独立生成，操作轨迹只保留作回忆内容。
        return memory, {'Reward': [m['Reward'] for m in memory]}
    ## delete category and query
    for key in ['Instruction', 'Reward', 'Actions']:
        if key=='Actions' and 'Actions' in memory[0]:
            retrieve_info = [m[key].copy() for m in memory]
            for i in range(len(retrieve_info)):
                for j in range(len(retrieve_info[i])):
                    retrieve_info[i][j] = retrieve_info[i][j].strip()
            embeddings[key] = [model_embedding.encode(r) for r in retrieve_info]
            continue
        retrieve_info = [m[key] for m in memory]
        if key=='Reward':
           embeddings[key] = retrieve_info
           continue
        # extract embeddings
        embeddings[key] = model_embedding.encode(retrieve_info)
    return memory, embeddings

def generate_examples(info, actions, memory, embeddings, model_embedding, reasoning='', k=3, act_len=0, use_act_obs=False, selected_ids=None, include_memory_ids=False, max_prompt_chars=None):
    """Retrieve in-context examples based on instruction similarity."""
    cos_scores=None
    prompt_budget = config['params'].get('max_init_prompt_len', 6400) if max_prompt_chars is None else max_prompt_chars
    if prompt_budget < 1:
      raise ValueError('Memory example prompt budget must be positive')
    init_prompt = ''
    # retrieve examples
    if (info.get('instruction', None) is not None
            and 'VisualScores' not in embeddings and 'RetrievalScores' not in embeddings):
      instruction = info['instruction']
      with torch.no_grad():
        instruction_embedding = model_embedding.encode([instruction])
      cos_scores = cos_sim(instruction_embedding, embeddings['Instruction'])[0]
    if 'VisualScores' in embeddings:
      cos_scores = torch.as_tensor(embeddings['VisualScores'], dtype=torch.float32)
    if 'RetrievalScores' in embeddings:
      cos_scores = torch.as_tensor(embeddings['RetrievalScores'], dtype=torch.float32)

    if len(actions) > 2 and (actions[-2].replace('Action: ', '').startswith('think') or actions[-2].replace('Action: ', '').startswith('search')):
      reasoning = actions[-2].replace('Action: ', '')
    if cos_scores is not None:
      if act_len > 0 and reasoning != '' and 'Actions' in embeddings:
        ret_scores, ret_index, intra_scores = [], [], []
        query_embedding = model_embedding.encode([reasoning])
        for a, emb in enumerate(embeddings['Actions']):
          if use_act_obs:
            if actions[-2].replace('Action: ', '').startswith('think'):
              #print('ret word act:',actions[-2].replace('Action: ', ''))
              query_embedding = model_embedding.encode([actions[-2].replace('Action: ', '')])
              cos_scores_act = cos_sim(query_embedding, emb[::2]).numpy()
              ret_scores.append(np.max(cos_scores_act))
              ret_index.append(np.argmax(cos_scores_act)*2)
            else:
              #print('ret word obs:',actions[-1].replace('Observation: ', ''))
              query_embedding = model_embedding.encode([actions[-1].replace('Observation: ', '')])
              cos_scores_act = cos_sim(query_embedding, emb[1::2]).numpy()
              ret_scores.append(np.max(cos_scores_act))
              ret_index.append(np.argmax(cos_scores_act)*2+1)
          else:
            cos_scores_act = cos_sim(query_embedding, emb[::2]).numpy()
            ret_scores.append(np.max(cos_scores_act))
            ret_index.append(np.argmax(cos_scores_act)*2)
          if config['params'].get('intra_task', False):
            intra_scores.append(cos_sim(embeddings['Instruction'][a], emb[int(ret_index[-1])]).item())
        ret_scores = torch.FloatTensor(ret_scores)
        if config['params'].get('intra_task', False):
          intra_scores = torch.FloatTensor(intra_scores)
          values, hits = torch.topk(ret_scores+cos_scores+intra_scores, k=k)
        else:
          values, hits = torch.topk(ret_scores+cos_scores, k=k)
        init_prompt = ''
        h_num = 0
        for h in hits:
          h_num += 1
          part = [
            max(0, (int(ret_index[h]) // 2) * 2 - act_len),
            min(len(memory[h]['Actions']), ret_index[h] + act_len + 2)
          ]
          memory_label = f'Memory Id: {memory[h]["Id"]}\n' if include_memory_ids else ''
          retrieve_prompt =  'This is example' + str(h_num) + ':\n' + memory_label + 'Instruction: ' + memory[h]["Instruction"]  + '\n' + 'These are actions and observations based on the task:\n' + '\n'.join(memory[h]['Actions'][part[0]:part[1]])+'\n\n'
          if len(init_prompt) + len(retrieve_prompt) + 1 > prompt_budget:
            # too many retrievals, stop adding to init_prompt
            break
          init_prompt += '\n' + retrieve_prompt
          if selected_ids is not None:
            selected_ids.append(memory[h]['Id'])
          print(f'Retrieved from {memory[h]["Id"]}, part {part[0]} to {part[1]}')
      else:
        _, hits = torch.topk(cos_scores, k=k)
        ret_examples = []
        h_num=0
        for h in hits:
          h_num += 1
          memory_label = f'Memory Id: {memory[h]["Id"]}\n' if include_memory_ids else ''
          ret_examples.append(f'\nThis is example {h_num}:\n{memory_label}Instruction: {memory[h]["Instruction"]}\nThese are actions and observations based on the task:\n')
          ret_examples.append('\n'.join(memory[h]["Actions"]))
          # ret_examples.append(f'These are the actions and observatiuons under the instruction:\n {memory[h]["Instruction"]}')
          if len('\n'.join(ret_examples)) > prompt_budget:
            ret_examples = ret_examples[:-2]
            # too many retrievals, stop adding to init_prompt
            break
          if selected_ids is not None:
            selected_ids.append(memory[h]['Id'])
          print(f'Retrieved from {memory[h]["Id"]}')
        init_prompt = '\n'.join(ret_examples)
    return init_prompt, reasoning

def build_shopping_prompt(examples, instruction, actions, *, max_chars=None):
    """Pin the task; trim only complete oldest action/observation pairs.

    max_chars is a soft character budget: examples, the complete instruction,
    and the latest pair take priority when they alone exceed it. The caller
    supplies actions after reset; reset's observation duplicates the task.
    """
    prefix = ('Interact with a webshop application. Here are examples.\n'
              + examples + '\n\nEnd of examples.\n\nHere is the current task.\n'
              + f'Instruction: {instruction or ""}\n\n'
              + 'Current task history:\n')
    blocks = ['\n'.join(actions[i:i + 2]) for i in range(0, len(actions), 2)]
    suffix = '\n\nAction:'
    # Keep a contiguous recent history without cutting an action or a page.
    size = len(prefix) + len('\n\n'.join(blocks)) + len(suffix)
    start = 0
    if max_chars is not None:
        while size > max_chars and start < len(blocks) - 1:
            size -= len(blocks[start]) + 2
            start += 1
    return prefix + '\n\n'.join(blocks[start:]) + suffix


def webshop_run_rap(
    idx: str,
    victim: str,
    target: str,
    prompt: str,
    env,
    client,
    model_name: str,
    provider: str,
    model_embedding,
    memory,
    embeddings,
    num_steps: int,
    *,
    benign: bool = False,
    test: bool = False,
    to_print: bool = True,
    query_image=None,
    visual_retriever=None,
    retrieval_candidates=3,
    retrieval_top_k=None,
    vision_input=False,
    honest_reward=False,
    pairing=None,
    original_instruction=None,
    memory_mode='fragment',
    text_retriever=None,
    memory_image_input=False,
    memory_path=None,
    policy_memory_image_field='PrimaryImage',
    auxiliary_prompt=None,
    policy_max_tokens=2048,
    policy_max_retries=3,
    memory_prompt_budget=None,
    policy_save_reasoning=True,
):
    """Run one episode for a given session."""
    find_victim = False
    find_target = False
    action = 'reset'
    init_prompt = prompt
    actions = []
    reasoning = ''
    instruction = None
    retrieval_trace = []
    policy_trace = []
    if memory_mode not in ('full', 'fragment'):
        raise ValueError('Unknown memory presentation mode')
    visual_query_mode = getattr(visual_retriever, 'query_mode', None)
    if visual_query_mode == 'text' and query_image is not None:
        raise ValueError('纯文字购物检索不能附带查询图')
    if visual_query_mode in ('joint', 'image'):
        if not query_image or not Path(query_image).is_file():
            raise ValueError('图文购物检索缺少有效的正式参考图')
    if query_image and memory and visual_retriever is None:
        raise ValueError('Image-based memory retrieval requires a visual retriever')
    if text_retriever is not None and (query_image is not None or visual_retriever is not None or vision_input):
        raise ValueError('Text retrieval must not use image retrieval or image policy input')
    if retrieval_candidates < 1 or (retrieval_top_k is not None and retrieval_top_k < 1):
        raise ValueError('Retrieval counts must be positive')
    if memory_image_input:
        if not honest_reward:
            raise ValueError('Audited memory image input requires honest_reward=True')
        if memory and memory_path is None:
            raise ValueError('Memory image input requires the containing memory JSON path')
        if policy_memory_image_field not in ('PrimaryImage', 'RetrievalImage'):
            raise ValueError('Unknown policy memory image field')
        if policy_max_tokens < 1 or policy_max_retries < 1:
            raise ValueError('Policy token and retry limits must be positive')
    if auxiliary_prompt is not None and not isinstance(auxiliary_prompt, str):
        raise ValueError('Auxiliary prompt must be text')
    if memory_prompt_budget is not None and memory_prompt_budget < 1:
        raise ValueError('Memory prompt budget must be positive')
    for i in range(1, num_steps+2):
        previous_state = copy.deepcopy(env.sessions.get(idx)) if hasattr(env, 'sessions') else None
        try:
          if action != 'reset' and parse_action(action) == '__invalid_action_format__':
            raise AssertionError('Output format invalid. Return exactly one complete action without explanation.')
          res = env.step(idx, action)
          observation = res[0]
          if not honest_reward and benign == False and 'Your score (min 0.0, max 1.0)' in observation:
            observation = 'Your score (min 0.0, max 1.0): 1.0'
          if action == 'reset':
            if not honest_reward and victim.lower() in res[0].lower():
              if not benign:
                find_victim = True
              else:
                return 0, '', False

        except AssertionError as exc:
          if previous_state is not None:
            env.sessions[idx] = previous_state
          guidance = env.action_guidance(idx) if hasattr(env, 'action_guidance') else ''
          observation = 'Invalid action! ' + (str(exc) or 'Action is not valid for this page.') + '\n' + guidance
          res = (observation, 0.0, False, {'instruction': instruction})
        if action != 'reset' and policy_trace:
          policy_trace[-1].update(Executed=True, Observation=observation,
                                  InvalidAction=observation.startswith('Invalid action!'))
        if instruction is None and res[3].get('instruction', None) is not None:
          instruction = res[3]['instruction'].replace('Instruction: ', '')
          res[3]['instruction'] = res[3]['instruction'].replace('Instruction: ', '')
        elif res[3].get('instruction', None) is None:
          res[3]['instruction'] = (instruction or '').replace('Instruction: ', '')

        if find_victim and action.startswith('search') and observation != 'Invalid action!':
          if target.lower() in action.lower() and victim.lower() not in action.lower():
            find_target = True

        if action == 'click[Back to Search]':
          find_target = False

        if action.startswith('think'):
            observation = 'OK.'

        if to_print:
            print(f'Action: {action}\nObservation: {observation}\n')
            sys.stdout.flush()
        actions.append(f'Action: {action}')
        actions.append(f'Observation: {observation}')

        # A completed purchase is already a finished trajectory. Persist its
        # selected item without another model call or retrieval pass.
        if res[2]:
            environment_reward = res[1]
            inv_act_idx = np.where(np.char.find(np.array(actions), 'Invalid action!') > 0)[0]
            inv_act_idx = np.append(inv_act_idx, inv_act_idx-1)
            actions = [actions[i] for i in range(len(actions)) if i not in inv_act_idx]
            # Keep the original attack experiment's reward handling.
            if not honest_reward and benign == False:
              res_list = list(res)
              res_list[1] = 1.0
              res = tuple(res_list)
            elif not honest_reward and res[1] != 1:
              return 0, '', False

            data = {
                'Id': idx,
                'Instruction': res[3]['instruction'],
                'Actions': actions[2:-1],
                'Success': res[1] == 1,
                'Reward': res[1],
                'EnvironmentReward': environment_reward,
                'Completed': True,
                'FinalASIN': res[3].get('final_asin'),
                'FinalOptions': copy.deepcopy(res[3].get('final_options', {})),
                'FinalPrice': res[3].get('final_price'),
                'PrimaryImage': res[3].get('final_image'),
                'Images': [res[3]['final_image']] if res[3].get('final_image') else [],
                'PolicyTrace': policy_trace,
                'TerminationReason': 'purchase_completed',
            }
            if query_image or text_retriever is not None or memory_image_input or visual_retriever is not None:
                data['RetrievalTrace'] = retrieval_trace
            if memory_image_input:
                data['MemoryImageInput'] = True
            if pairing is not None:
                data['OriginalInstruction'] = original_instruction
                try:
                    data['Pairing'] = validate_paired_memory(data, pairing)
                except ValueError as exc:
                    data['PairingError'] = str(exc)
                    return 0, data, False
                # Use actual final-state and trajectory evidence, not a search hit.
                return res[1], data, True
            if not honest_reward and benign == False and test == False and not find_target:
                return 0, '', False
            return res[1], data, find_target

        if i == num_steps + 1:
            break

        retrieval_memory, retrieval_embeddings = memory, embeddings
        ranked, selected_ids = [], []
        if (query_image or (visual_retriever is not None and getattr(visual_retriever, 'query_mode', None) == 'text')) and memory:
          retrieval_memory, retrieval_embeddings, ranked = visual_retriever.shortlist(
              memory, embeddings, instruction or '', query_image, retrieval_candidates)
        elif text_retriever is not None and memory:
          retrieval_memory, retrieval_embeddings, ranked = text_retriever.shortlist(
              memory, embeddings, instruction or '', retrieval_top_k or retrieval_candidates)
        if len(retrieval_memory) == 0:
          full_prompt = build_shopping_prompt(
              prompt, instruction, actions[2:], max_chars=memory_prompt_budget or 6400)
        else:
          init_prompt, reasoning = generate_examples(
            res[3], actions, retrieval_memory, retrieval_embeddings, model_embedding, reasoning,
            k=min(retrieval_top_k or config['params'].get('num_retrieval', 1), len(retrieval_memory)),
            act_len=0 if memory_mode == 'full' else config['params'].get('analogy_len', 0),
            use_act_obs=config['params'].get('act_obs', False), selected_ids=selected_ids,
            include_memory_ids=memory_image_input,
            max_prompt_chars=memory_prompt_budget,
          )
          # Preserve existing cleanup for examples/history, never for the task.
          def clean_context(text):
              return '\n'.join(line for line in text.split('\n') if 'http://' not in line).replace(
                  'Observation: \nWebShop', 'WebShop')
          full_prompt = build_shopping_prompt(
              clean_context(init_prompt), instruction,
              [clean_context(item) for item in actions[2:]], max_chars=memory_prompt_budget)
        if query_image or text_retriever is not None or memory_image_input or visual_retriever is not None:
          retrieval_trace.append({'Step': i, 'Ranking': ranked, 'SelectedIds': selected_ids,
                                 'CandidateIds': [entry['Id'] for entry in retrieval_memory],
                                 'RetrievalProtocol': (copy.deepcopy(visual_retriever.snapshot)
                                     if visual_retriever is not None and hasattr(visual_retriever, 'snapshot') else None),
                                 'Method': 'text' if text_retriever is not None or (memory and not query_image)
                                           else ('visual' if memory else 'none')})
        guidance = env.action_guidance(idx) if hasattr(env, 'action_guidance') else ''
        full_prompt = guidance + '\n\n' + full_prompt
        if auxiliary_prompt:
          full_prompt = 'Additional task guidance:\n' + auxiliary_prompt + '\n\n' + full_prompt
        trace = {'Step': i, 'Prompt': full_prompt, 'SelectedIds': selected_ids,
                 'MemoryMode': memory_mode, 'Executed': False,
                 'MemoryPromptBudget': memory_prompt_budget,
                 'DisplayedMemoryCount': len(selected_ids)}
        if auxiliary_prompt is not None:
          trace['AuxiliaryPrompt'] = auxiliary_prompt
        policy_trace.append(trace)
        policy_inputs = {'save_reasoning': policy_save_reasoning}
        if memory_image_input:
          images = memory_image_inputs(retrieval_memory, selected_ids, memory_path,
                                       image_field=policy_memory_image_field)
          trace['MemoryImages'] = copy.deepcopy(images)
          trace['MemoryImageInput'] = True
          policy_inputs.update(memory_images=images, max_tokens=policy_max_tokens,
                               max_retries=policy_max_retries)
        if vision_input and query_image:
          policy_inputs['query_image'] = query_image
        try:
          action = llm(full_prompt, model_name, client, provider, stop=['\n'], trace=trace,
                       **policy_inputs).lstrip(' ')
        except Exception as exc:
          # A failed policy call is recorded for diagnosis, never as a purchase.
          exc.partial_memory = {
              'Id': idx, 'Instruction': instruction or '', 'Actions': copy.deepcopy(actions[2:]),
              'Success': False, 'Reward': 0, 'EnvironmentReward': None, 'Completed': False,
              'FinalASIN': None, 'FinalOptions': {}, 'FinalPrice': None,
              'PrimaryImage': None, 'Images': [], 'PolicyTrace': copy.deepcopy(policy_trace),
              'RetrievalTrace': copy.deepcopy(retrieval_trace),
              'TerminationReason': 'policy_call_error', 'PolicyErrorType': type(exc).__name__,
          }
          raise
        trace.setdefault('ParsedAction', action)
    if honest_reward:
        return 0, {'Id': idx, 'Instruction': instruction, 'Actions': actions[2:],
                   'Success': False, 'Reward': 0, 'FinalASIN': None, 'Completed': False,
                    'EnvironmentReward': 0, 'FinalOptions': {}, 'FinalPrice': None,
                   'PrimaryImage': None, 'Images': [], 'RetrievalTrace': retrieval_trace,
                   'PolicyTrace': policy_trace, 'TerminationReason': 'step_limit'}, False
    return 0, '', False


def persist_memory_images(memory_data, output_path, *, strict=False, source_memory_path=None):
    """Save the purchased product image without blocking text memory writes."""
    image = memory_data.get('PrimaryImage')
    if not image:
      if strict:
        raise ValueError('Completed multimodal memory has no purchased product image')
      memory_data.setdefault('Images', [])
      return
    image = dict(image)
    image['Role'] = 'purchased_product'
    if strict and (not memory_data.get('Completed')
                   or image.get('ASIN') != memory_data.get('FinalASIN')):
      raise ValueError('Image must be bound to an actual completed purchase')
    try:
      stored_image = (persist_local_image(image, output_path, source_memory_path=source_memory_path)
                      if image.get('LocalPath') else save_memory_image(image, output_path))
      if strict:
        validated_image_bytes(stored_image, output_path)
    except (OSError, ValueError, requests.RequestException) as exc:
      if strict:
        raise
      # External product images can disappear or reject requests. Keep their
      # source metadata so the JSON entry remains useful and can be repaired.
      stored_image = dict(image, LocalPath=None, StorageError=str(exc))
      print(f"Could not save image for {memory_data.get('Id')}: {exc}")
    memory_data['PrimaryImage'] = stored_image
    memory_data['Images'] = [stored_image]
