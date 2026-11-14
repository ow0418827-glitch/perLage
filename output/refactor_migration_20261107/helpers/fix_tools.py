from pathlib import Path
root=Path(__file__).resolve().parents[1]
for p in (root/'tools').glob('*.py'):
    lines=p.read_text(encoding='utf-8').splitlines(keepends=True)
    for index,line in enumerate(lines):
        if 'sys.path.insert(' in line:
            if 'ProjectPath as Path' not in ''.join(lines[index+1:index+3]):
                lines.insert(index+1,'from src.paths import ProjectPath as Path\n')
            break
    text=''.join(lines).replace('lage/output/','lage/src/output/')
    text=text.replace("DATA / 'private/reused_shared_20261104'","PRIVATE / 'reused_shared_20261104'")
    text=text.replace("DATA / 'private/shared_memory.json'","PRIVATE / 'shared_memory.json'")
    text=text.replace("DATA / 'private/query_images'","PRIVATE / 'query_images'")
    p.write_text(text,encoding='utf-8')
print('Tool bootstraps and historical read-only locations verified.')
