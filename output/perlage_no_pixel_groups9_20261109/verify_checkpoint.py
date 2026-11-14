import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.visual.image_utils import sha256_file
from src.visual.audit import read_json
from src.visual.attack_completion import validate_attack_completion

base = Path("output/perlage_no_pixel_groups9_20261109/groups")
groups = ["01_camera", "02_chair", "03_cookies", "04_curtain", "05_hair_extension"]

print("=" * 60)
print("检查前五组断点产物与冻结凭证：")
for g in groups:
    mroot = base / g / "methods" / "PerLAGE_no_pixel"
    f_json = mroot / "FROZEN_IMAGE.json"
    f_png = mroot / "frozen.png"
    assert f_json.is_file(), f"{g} 缺失 FROZEN_IMAGE.json"
    assert f_png.is_file(), f"{g} 缺失 frozen.png"
    data = read_json(f_json)
    h = sha256_file(f_png)
    assert h == data["image_sha256"], f"{g} 哈希不匹配"
    validate_attack_completion(data["metadata"], data.get("requested_steps", 60), {
        "method": "PerLAGE_no_pixel",
        "source_image_sha256": data["source"]["image_sha256"],
        "target_image_sha256": data["target"]["image_sha256"],
    })
    print(f"  [OK] {g}: SHA256={h[:16]}... 状态={data['status']}")

print("前五组全部 100% 审计通过，断点完美封存！")
print("=" * 60)
