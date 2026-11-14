"""Read frozen images only; calculate full-reference image metrics at native size."""
from pathlib import Path
import hashlib
import json

import numpy as np
from PIL import Image, __version__ as pillow_version
import skimage
from skimage.metrics import structural_similarity, peak_signal_noise_ratio


PROJECT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BASE = PROJECT / "output/experiment_base_groups9_20261109/groups"
RUNS = {
    "perlage_no_pixel": ("perlage_no_pixel_groups9_20261109", "PerLAGE_no_pixel"),
    "lucid": ("lucid_groups9_20261109", "lucid"),
}


def load(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    rows = []
    for group in sorted(BASE.iterdir()):
        if not group.is_dir():
            continue
        source_path = group / "source.jpg"
        if not source_path.is_file():
            continue
        source = load(source_path)
        row = {
            "group": group.name,
            "source": str(source_path),
            "source_sha256": sha(source_path),
            "shape": list(source.shape),
            "methods": {},
        }
        for name, (run, method) in RUNS.items():
            path = PROJECT / "output" / run / "groups" / group.name / "methods" / method / "frozen.png"
            image = load(path)
            if image.shape != source.shape:
                raise ValueError(f"Shape differs: {group.name} {name}")
            delta = image.astype(np.float64) - source.astype(np.float64)
            row["methods"][name] = {
                "path": str(path),
                "sha256": sha(path),
                "ssim": float(structural_similarity(
                    source, image, data_range=255, channel_axis=-1,
                    gaussian_weights=True, sigma=1.5, win_size=11,
                    use_sample_covariance=False,
                )),
                "mae": float(np.abs(delta).mean()),
                "psnr_db": float(peak_signal_noise_ratio(source, image, data_range=255)),
                "rmse": float(np.sqrt(np.mean(delta * delta))),
            }
        rows.append(row)
    summary = {}
    for name in RUNS:
        summary[name] = {
            "groups": len(rows),
            **{metric: float(np.mean([r["methods"][name][metric] for r in rows]))
               for metric in ("ssim", "mae", "psnr_db", "rmse")},
        }
    summary["perlage_wins"] = {
        "ssim_higher": sum(r["methods"]["perlage_no_pixel"]["ssim"] > r["methods"]["lucid"]["ssim"] for r in rows),
        "mae_lower": sum(r["methods"]["perlage_no_pixel"]["mae"] < r["methods"]["lucid"]["mae"] for r in rows),
        "psnr_higher": sum(r["methods"]["perlage_no_pixel"]["psnr_db"] > r["methods"]["lucid"]["psnr_db"] for r in rows),
    }
    audit = {
        "scope": "Post hoc full-reference image metrics from frozen files only; no model inference or image mutation.",
        "protocol": {
            "decoder": "PIL.Image.open().convert('RGB')",
            "pillow_version": pillow_version,
            "skimage_version": skimage.__version__,
            "numpy_version": np.__version__,
            "size": "native dimensions, no resampling or registration",
            "data_range": 255,
            "ssim": {"channel_axis": -1, "gaussian_weights": True, "sigma": 1.5, "win_size": 11, "use_sample_covariance": False},
            "aggregation": "arithmetic mean across nine groups; PSNR is mean of per-image dB values",
        },
        "limitations": [
            "Source-image similarity is not a human naturalness, perceptual invisibility or label-readability score.",
            "Whole-image metrics may be dominated by background; no foreground or text-region-specific metric is claimed.",
            "These post hoc diagnostics do not replace preregistered attack metrics, human blinded ratings or defended evaluation.",
        ],
        "summary": summary,
        "groups": rows,
    }
    (OUT / "image_quality_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "groups": [{"group":r["group"], "metrics":{k:{m:v[m] for m in ('ssim','mae','psnr_db')} for k,v in r['methods'].items()}} for r in rows]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
