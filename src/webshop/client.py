"""商城页面解析与动作环境客户端；模型推理留在智能体策略中。"""
import copy
import json
import re
from src.paths import ProjectPath as Path
from src.agent.memory.multimodal_memory import (
    extract_page_images, save_memory_image, snapshot_product_image, validated_image_bytes)

import requests
from bs4 import BeautifulSoup
from bs4.element import Comment

ACTION_TO_TEMPLATE = {
    'Description': 'description_page.html',
    'Features': 'features_page.html',
    'Reviews': 'review_page.html',
    'Attributes': 'attributes_page.html',
}
DEFAULT_WEBSHOP_URL = "http://localhost:3000/"

def clean_str(p):
  """Normalize escaped text content from HTML pages."""
  return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")


def tag_visible(element):
    ignore = {'style', 'script', 'head', 'title', 'meta', '[document]'}
    return (
        element.parent.name not in ignore and not isinstance(element, Comment)
    )


def webshop_text(session, page_type, query_string='', page_num=1, asin='', options=None, subpage='', instruction_override=None, http_timeout=(5, 30), base_url=DEFAULT_WEBSHOP_URL, **kwargs):
    """Fetch and parse WebShop text for a given session and page."""
    selected_options = dict(options or {})
    base = base_url.rstrip('/')
    if page_type == 'init':
      endpoint, params = f'{base}/{quote(str(session), safe="")}', None
    elif page_type in ('search', 'item', 'item_sub', 'end'):
      # The source /rap/page adapter delegates to its original render/reward
      # functions. Query parameters preserve slashes, hashes, quotes and Unicode.
      endpoint = base + '/rap/page'
      params = {'session_id': session, 'page_type': page_type}
      if page_type in ('search', 'item', 'item_sub'):
        params.update(keywords=query_string, page=page_num)
      if page_type in ('item', 'item_sub', 'end'):
        params.update(asin=asin, options=repr(selected_options))
      if page_type == 'item_sub':
        params['sub_page'] = subpage
    else:
      raise ValueError('Unknown WebShop page type: ' + str(page_type))
    response = requests.get(endpoint, params=params, timeout=http_timeout)
    response.raise_for_status()
    url = getattr(response, 'url', None) or requests.Request('GET', endpoint, params=params).prepare().url
    html = response.text
    html_obj = BeautifulSoup(html, 'html.parser')
    texts = html_obj.find_all(string=True)
    visible_texts = list(filter(tag_visible, texts))
    # visible_texts = [str(text).strip().strip('\\n') for text in visible_texts]
    # if page_type == 'end': import pdb; pdb.set_trace()
    if False:
        # For `simple` mode, return just [SEP] separators
        return ' [SEP] '.join(t.strip() for t in visible_texts if t != '\n')
    else:
        # Otherwise, return an observation with tags mapped to specific, unique separators
        observation = ''
        option_type = ''
        file_name = ''
        options = {}
        asins = []
        cnt = 0
        prod_cnt = 0
        just_prod = 0
        for t in visible_texts:
            if t == '\n': continue
            if t.replace('\n', '').replace('\\n', '').replace(' ', '') == '': continue
            # if t.startswith('Instruction:') and page_type != 'init': continue
            if t.parent.name == 'button':  # button
                processed_t = f'\n[{t}] '
            elif t.parent.name == 'label':  # options
                if str(t) in selected_options.values():
                    processed_t = f'[[{t}]]'
                    # observation = f'You have clicked {t}.\n' + observation
                else:
                    processed_t = f'[{t}]'
                options[str(t)] = option_type
                # options[option_type] = options.get(option_type, []) + [str(t)]
            elif t.parent.get('class') == ["product-link"]: # product asins
                processed_t = f'\n[{t}] '
                prod_cnt += 1
                asins.append(str(t))
                just_prod = 0
            else: # regular, unclickable text
                processed_t =  '\n' + str(t) + ' '
                if cnt < 2 and page_type != 'init': processed_t = ''
                option_type = str(t)
                cnt += 1
            just_prod += 1
            observation += processed_t
        info = {}
        info['has_next'] = any(str(t).strip() == 'Next >' for t in visible_texts)
        if options:
          info['option_types'] = options
        if asins:
          info['asins'] = asins
        if 'Your score (min 0.0, max 1.0)' in visible_texts:
          idx = visible_texts.index('Your score (min 0.0, max 1.0)')
          info['reward'] = float(visible_texts[idx + 1])
          observation = 'Your score (min 0.0, max 1.0): ' + (visible_texts[idx + 1])
          # observation = 'Your score (min 0.0, max 1.0): 1.0'
        # Keep each product image tied to its ASIN instead of storing HTML tags.
        if page_type in ('search', 'item'):
          info['images'] = extract_page_images(html_obj, page_type, asin=asin, page_url=url)
          if page_type == 'item' and info['images']:
            info['current_item_image'] = info['images'][0]
        if page_type == 'item':
          price_match = re.search(r'\bPrice:\s*\$?\s*([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)',
                                  '\n'.join(str(t) for t in visible_texts))
          info['current_item_price'] = float(price_match[1].replace(',', '')) if price_match else None
        # Get starting instruction text

        instruction = instruction_override
        info['instruction'] = instruction #if instruction is not None else ''
        observation = clean_str(observation)
        if instruction is not None and "Instruction: " in observation:
            observation = f"Webshop\nInstruction:\n{instruction}\n[Search]"
        return observation, info


from urllib.parse import quote
class webshopEnv:
  def __init__(self, instruction_overrides=None, *, image_snapshot_path=None, http_timeout=(5, 30),
               image_cache_root=None, image_timeout=(10, 30), image_max_attempts=2, base_url=DEFAULT_WEBSHOP_URL):
    self.sessions = {}
    self.base_url = base_url
    self.instruction_overrides = instruction_overrides or {}
    self.image_snapshot_path = Path(image_snapshot_path).resolve() if image_snapshot_path else None
    self.http_timeout = http_timeout
    self.image_cache_root = Path(image_cache_root).resolve() if image_cache_root else None
    self.image_timeout = image_timeout
    self.image_max_attempts = image_max_attempts
    self._episode_image_snapshots = {}

  def action_guidance(self, session):
    state = self.sessions.get(session, {})
    page = state.get('page_type', 'init')
    rules = ['Return exactly one action: think[reason], search[keywords], or click[visible button or product ID]. No prose or multiple actions.']
    if page == 'init':
      rules.append('You are on the search home page. You can search[keywords].')
    elif page == 'search':
      rules.append('You are on search results. To change keywords, first click[Back to Search], then search on the next turn. Do not search directly here.')
      rules.append('Product IDs available: ' + ', '.join(state.get('asins', [])))
      if state.get('has_next'):
        rules.append('You may click[Next >].')
      if state.get('page_num', 1) > 1:
        rules.append('You may click[< Prev].')
    elif page == 'item':
      rules.append('You are on a product page. You may select a displayed option, click[Description], click[Features], click[Reviews], click[Attributes], click[Buy Now], click[< Prev], or click[Back to Search].')
      rules.append('Options: ' + ', '.join(state.get('option_types', {})))
    elif page == 'item_sub':
      rules.append('You are on a product information subpage. Use click[< Prev] to return to the product, or click[Back to Search] to search again.')
    return '\n'.join(rules)

  def step(self, session, action):
    done = False
    observation_ = None
    step_index = self.sessions.get(session, {}).get('step_index', 0) + 1
    if action == 'reset':
      self.sessions[session] = {'session': session, 'page_type': 'init'}
      self._episode_image_snapshots[session] = {}
      step_index = 1
    elif action.startswith('think['):
      observation = 'OK.'
    elif action.startswith('search['):
      assert self.sessions[session]['page_type'] == 'init', 'Search requires the home page. First click[Back to Search].'
      query = action[7:-1]
      self.sessions[session] = {'session': session, 'page_type': 'search',
                                'query_string': query, 'page_num': 1}
    elif action.startswith('click['):
      button = action[6:-1]
      if button == 'Buy Now':
        assert self.sessions[session]['page_type'] == 'item'
        # The done page has no product image; freeze the selected item first.
        final_asin = self.sessions[session].get('asin')
        current_image = self.sessions[session].get('current_item_image')
        self.sessions[session]['final_asin'] = final_asin
        self.sessions[session]['final_options'] = copy.deepcopy(self.sessions[session].get('options', {}))
        self.sessions[session]['final_price'] = self.sessions[session].get('current_item_price')
        self.sessions[session]['final_image'] = (
            dict(current_image) if current_image and current_image.get('ASIN') == final_asin else None
        )
        # /rap/page transports original options safely; rewards need raw values.
        self.sessions[session]['page_type'] = 'end'
        done = True
      elif button == 'Back to Search':
        assert self.sessions[session]['page_type'] in ['search', 'item_sub', 'item']
        self.sessions[session] = {'session': session, 'page_type': 'init'}
      elif button == 'Next >':
        assert self.sessions[session]['page_type'] == 'search'
        assert self.sessions[session].get('has_next', False), 'There is no next page.'
        self.sessions[session]['page_num'] += 1
      elif button == '< Prev':
        assert self.sessions[session]['page_type'] in ['search', 'item_sub', 'item']
        if self.sessions[session]['page_type'] == 'search':
          assert self.sessions[session]['page_num'] > 1, 'Already on the first results page.'
          self.sessions[session]['page_num'] -= 1
        elif self.sessions[session]['page_type'] == 'item_sub':
          self.sessions[session]['page_type'] = 'item'
        elif self.sessions[session]['page_type'] == 'item':
          self.sessions[session]['page_type'] = 'search'
          self.sessions[session]['options'] = {}
          self.sessions[session].pop('current_item_image', None)
          self.sessions[session].pop('current_item_price', None)
      elif button in ACTION_TO_TEMPLATE:
        assert self.sessions[session]['page_type'] == 'item'
        self.sessions[session]['page_type'] = 'item_sub'
        self.sessions[session]['subpage'] = button
      else:
        if self.sessions[session]['page_type'] == 'search':
          assert button in self.sessions[session].get('asins', [])  # must be asins
          self.sessions[session]['page_type'] = 'item'
          self.sessions[session]['asin'] = button
        elif self.sessions[session]['page_type'] == 'item':
          assert 'option_types' in self.sessions[session]
          assert button in self.sessions[session]['option_types'], (button, self.sessions[session]['option_types'])  # must be options
          option_type = self.sessions[session]['option_types'][button]
          if not 'options' in self.sessions[session]:
            self.sessions[session]['options'] = {}
          self.sessions[session]['options'][option_type] = button
          observation_ = f'You have clicked {button}.'
        else:
          raise AssertionError('Use a button valid for the current page.')
    else:
      assert False
    self.sessions[session]['step_index'] = step_index
    observation, info = webshop_text(**self.sessions[session],
                                    instruction_override=self.instruction_overrides.get(session),
                                    http_timeout=self.http_timeout, base_url=self.base_url)
    if observation_:
      observation = observation_
    if info.get('current_item_image'):
      info['current_item_image']['Step'] = step_index
      if self.image_snapshot_path is not None:
        observed_image = info['current_item_image']
        if observed_image.get('ASIN') != self.sessions[session].get('asin'):
          raise ValueError('Observed product image ASIN differs from the current item')
        key = (observed_image['ASIN'], observed_image.get('SourceURL'))
        episode_cache = self._episode_image_snapshots.setdefault(session, {})
        if key in episode_cache:
          snapshot = copy.deepcopy(episode_cache[key])
          validated_image_bytes(snapshot)
          snapshot['ImageSnapshotCacheHit'] = True
        else:
          if self.image_cache_root is not None:
            snapshot = snapshot_product_image(
                observed_image, self.image_snapshot_path, image_cache_root=self.image_cache_root,
                timeout=self.image_timeout, max_attempts=self.image_max_attempts)
          else:
            snapshot = save_memory_image(observed_image, self.image_snapshot_path)
          snapshot['LocalPath'] = str((self.image_snapshot_path.parent / snapshot['LocalPath']).resolve())
          snapshot.setdefault('OriginalURL', observed_image.get('SourceURL'))
          snapshot['ObservedSHA256'] = snapshot['SHA256']
          snapshot['FirstObservedStep'] = step_index
          snapshot['ImageSnapshotCacheHit'] = False
          episode_cache[key] = copy.deepcopy(snapshot)
        snapshot['LocalPath'] = str((self.image_snapshot_path.parent / snapshot['LocalPath']).resolve())
        snapshot['Step'] = step_index
        info['current_item_image'] = snapshot
        info['images'] = [copy.deepcopy(snapshot)]
    elif self.image_snapshot_path is not None and self.sessions[session].get('page_type') == 'item':
      raise ValueError('Product page has no image; audited multimodal episode cannot continue')
    if done:
      info['final_asin'] = self.sessions[session].get('final_asin')
      info['final_image'] = self.sessions[session].get('final_image')
      info['final_options'] = self.sessions[session].get('final_options', {})
      info['final_price'] = self.sessions[session].get('final_price')
    self.sessions[session].update(info)
    reward = info.get('reward', 0.0)

    return observation, reward, done, info


