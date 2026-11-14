from pathlib import Path
root = Path(__file__).resolve().parents[1]
for name in ['src/experiments/pipeline.py', 'src/experiments/public_pair_selection.py',
             'src/experiments/shared_batch.py', 'tools/run_shared_groups29.py']:
    p = root / name
    text = p.read_text(encoding='utf-8')
    text = text.replace("DATA / 'public'", 'PUBLIC').replace("DATA/'public'", 'PUBLIC')
    text = text.replace("DATA/'public/groups'", "PUBLIC/'groups'").replace("DATA / 'public/groups'", "PUBLIC / 'groups'")
    text = text.replace("DATA/'public/public_products_v2.sqlite'", "PUBLIC/'public_products_v2.sqlite'")
    text = text.replace("DATA/'public/selection_runs/simple_v2'", "PUBLIC/'selection_runs/simple_v2'")
    text = text.replace("DATA/'private/shared_memory_perlage.json'", "PRIVATE/'shared_memory_perlage.json'")
    text = text.replace('from src.paths import ', 'from src.paths import PUBLIC, PRIVATE, VISUAL_CONFIGS, EXPERIMENT_CONFIGS, public_root_for, ', 1)
    text = text.replace("CONFIGS / 'optimization.json'", "VISUAL_CONFIGS / 'optimization.json'")
    text = text.replace("CONFIGS / 'pipeline.example.json'", "EXPERIMENT_CONFIGS / 'pipeline.example.json'")
    text = text.replace("CONFIGS/'selection_groups29.example.json'", "EXPERIMENT_CONFIGS/'selection_groups29.example.json'")
    text = text.replace("'src/configs/optimization.json'", "'src/visual/config/optimization.json'")
    text = text.replace("('src/vision', 'src/vendor')", "('src/visual',)")
    p.write_text(text, encoding='utf-8')
p = root / 'tools/prepare_models.py'
text = p.read_text(encoding='utf-8').replace('MODEL_ROOT, CONFIGS', 'MODEL_ROOT, VISUAL_CONFIGS').replace("CONFIGS/", "VISUAL_CONFIGS/")
p.write_text(text, encoding='utf-8')
p = root / 'run_pipeline.ps1'
p.write_text(p.read_text(encoding='utf-8').replace('src/configs/pipeline.example.json','src/experiments/config/pipeline.example.json'), encoding='utf-8')
p = root / 'src/experiments/title_target.py'
p.write_text(p.read_text(encoding='utf-8').replace("'vision/dual_rag_bridge.py'", "'visual/_lucid/pipeline.py'"), encoding='utf-8')
