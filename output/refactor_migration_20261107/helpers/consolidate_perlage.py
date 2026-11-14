from pathlib import Path
root=Path(__file__).resolve().parents[1]
for old,new in [('src/visual/_natural','src/visual/_perlage/components'),
                ('src/visual/natural_visual_edit.py','src/visual/_perlage/components/instruction.py'),
                ('src/visual/latent_branch.py','src/visual/_perlage/components/latent_branch.py')]:
    source,destination=(root/old).resolve(),(root/new).resolve()
    if not source.is_relative_to(root) or not destination.is_relative_to(root):raise ValueError('Invalid relocation')
    source.rename(destination)
for path in [*(root/'src').rglob('*.py'),*(root/'tools').glob('*.py')]:
    if path.name=='paths.py':continue
    text=path.read_text(encoding='utf-8')
    text=text.replace('src.visual._natural','src.visual._perlage.components')
    text=text.replace('src.visual.natural_visual_edit','src.visual._perlage.components.instruction')
    text=text.replace('src.visual.latent_branch','src.visual._perlage.components.latent_branch')
    text=text.replace('src/visual/_natural/','src/visual/_perlage/components/')
    text=text.replace('src/visual/natural_visual_edit.py','src/visual/_perlage/components/instruction.py')
    text=text.replace('src/visual/latent_branch.py','src/visual/_perlage/components/latent_branch.py')
    path.write_text(text,encoding='utf-8')
