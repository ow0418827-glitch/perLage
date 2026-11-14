from pathlib import Path
root=Path(__file__).resolve().parents[1]
p=root/'src/visual/_perlage/components/core.py'
text=p.read_text(encoding='utf-8')
text=text[:text.index('\ndef public_development_value(')]+'\n'
start=text.index('from src.paths import relocate_io_path')
end=text.index('@dataclass(frozen=True)')
text=text[:start]+'''from dataclasses import asdict,dataclass
import itertools
import math
import torch
import torch.nn.functional as F

'''+text[end:]
text=text.replace('shadow_weight: float = 1.0','shadow_weight: float = 0.0')
p.write_text(text,encoding='utf-8')
p=root/'src/visual/_perlage/components/optimizer.py';text=p.read_text(encoding='utf-8')
text=text.replace(', public_development_value','')
text=text.replace('from src.visual.public_pool import margin_loss, PublicMarginConfig\n','')
text=text.replace("    if config.shadow_weight and public_bank is None:\n        raise ValueError('启用公开竞争目标需要可审计的公开池')\n",'')
text=text.replace("public_bank_sha256=public_bank['sha256'] if public_bank is not None else None",'shadow_pool_used=False')
text=text.replace("sequence='instruction_edit_then_latent_only_then_frozen_generation_plus_small_pixel'", "sequence='instruction_edit_then_latent_only' if config.pixel_clean_steps == config.pixel_diffusion_steps == 0 else 'instruction_edit_then_latent_and_pixel'")
text=text.replace('def utility(image, thresholds=None):','def utility(image):')
text=text.replace('        if thresholds is not None and config.shadow_weight:\n            value = value - config.shadow_weight * margin_loss(features, joint, target_feature, thresholds, PublicMarginConfig())\n','')
start=text.index('                    if public_bank is not None:')
end=text.index("                    row['diffusion'].append(item)",start)
text=text[:start]+"                    item['public_margin_value'] = item['utility']\n"+text[end:]
text=text.replace("            thresholds = public_bank['train_ordinary'] if public_bank is not None else None\n",'')
text=text.replace("                    thresholds = public_bank['train_diffusion'][str(timestep)] if public_bank is not None else None\n",'')
text=text.replace('utility(transformed_batch(candidate), thresholds)','utility(transformed_batch(candidate))')
text=text.replace('utility(purified, thresholds)','utility(purified)')
p.write_text(text,encoding='utf-8')
source=(root/'src/visual/public_pool.py').resolve();destination=(root/'src/deprecated/public_pool.py').resolve()
if not source.is_relative_to(root) or not destination.is_relative_to(root):raise ValueError('Invalid relocation')
source.rename(destination)
print('Unused competition pool removed from active generation dependency graph.')
