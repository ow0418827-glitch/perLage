import sys
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from transformers import Sam2Processor, Sam2Model

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import model_path

img_path = "output/experiment_base_groups9_20261109/groups/06_jeans/source.jpg"
boxes = [[90, 0, 311, 495]]
mpath = str(model_path("sam2.1-hiera-tiny-source"))

image = Image.open(img_path).convert("RGB")
processor = Sam2Processor.from_pretrained(mpath, local_files_only=True)
model = Sam2Model.from_pretrained(mpath, local_files_only=True)
inputs = processor(images=image, input_boxes=[boxes], return_tensors="pt")
with torch.no_grad():
    pred = model(**inputs, multimask_output=True)
masks = processor.post_process_masks(pred.pred_masks, inputs["original_sizes"])[0].cpu().numpy()
scores = pred.iou_scores[0].cpu().numpy()

print("Image size:", image.size)
print("Masks shape:", masks.shape)
print("Scores:", scores)
x = round((boxes[0][0] + boxes[0][2]) / 2)
y = round((boxes[0][1] + boxes[0][3]) / 2)
print(f"Center coords: ({x}, {y})")

for i in range(3):
    in_center = bool(masks[0, i, y, x])
    box_overlap = masks[0, i, boxes[0][1]:boxes[0][3], boxes[0][0]:boxes[0][2]].sum()
    print(f"Mask {i}: sum={masks[0, i].sum()}, score={scores[0, i]:.4f}, in_center={in_center}, overlap_in_box={box_overlap}")

# Also compare with ground truth source_foreground.png
gt_fg = np.asarray(Image.open("output/experiment_base_groups9_20261109/groups/06_jeans/source_foreground.png").convert("L")) > 0
print(f"Ground truth source_foreground.png sum: {gt_fg.sum()}, gt in center: {gt_fg[y, x]}")
for i in range(3):
    intersection = np.logical_and(masks[0, i], gt_fg).sum()
    union = np.logical_or(masks[0, i], gt_fg).sum()
    iou = intersection / union if union > 0 else 0
    print(f"  Mask {i} IoU with gt: {iou:.4f}")
