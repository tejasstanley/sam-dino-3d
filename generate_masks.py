# scripts/1_generate_masks.py

import os
import json
import cv2
import numpy as np
import torch
import random
from tqdm import tqdm

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


# ---------------- CONFIG ---------------- #

IMAGE_DIR = "data/images"
OUT_DIR   = "outputs/sam_masks"

SAM_CHECKPOINT = "sam_vit_l_0b3195.pth"
MODEL_TYPE = "vit_l"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MIN_AREA = 800
MAX_AREA_RATIO = 0.85
IOU_DUP_THRESH = 0.9


# ---------------- UTILS ---------------- #

def mask_iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / (union + 1e-6)


def remove_duplicate_masks(masks):
    keep = []
    for m in masks:
        is_dup = False
        for k in keep:
            if mask_iou(m, k) > IOU_DUP_THRESH:
                is_dup = True
                break
        if not is_dup:
            keep.append(m)
    return keep


def is_ground_like(mask, H):
    ys = np.where(mask)[0]
    if len(ys) == 0:
        return True
    return (ys.mean() > 0.7 * H)


# ---------------- OVERLAY VIS ---------------- #

def save_overlay(image_rgb, masks, save_path):
    overlay = image_rgb.copy()
    alpha = 0.5
    random.seed(0)

    for idx, mask in enumerate(masks):

        color = np.array([
            random.randint(40, 255),
            random.randint(40, 255),
            random.randint(40, 255)
        ], dtype=np.uint8)

        # fill
        overlay[mask] = (
            overlay[mask] * (1 - alpha) + color * alpha
        ).astype(np.uint8)

        # boundary
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 0, 0), 1)

        # centroid label
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        cx, cy = int(xs.mean()), int(ys.mean())

        cv2.putText(
            overlay, str(idx), (cx, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45, (255, 255, 255), 1, cv2.LINE_AA
        )

    # header
    header = f"Segments: {len(masks)}"
    (w, h), _ = cv2.getTextSize(header, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)

    cv2.rectangle(overlay, (0, 0), (w + 20, h + 20), (0, 0, 0), -1)
    cv2.putText(
        overlay, header, (10, h + 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
        (0, 255, 255), 2, cv2.LINE_AA
    )

    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, overlay_bgr)


# ---------------- LOAD SAM ---------------- #

print("Loading SAM...")

sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
sam.to(DEVICE)

mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    min_mask_region_area=MIN_AREA
)


# ---------------- MAIN ---------------- #

os.makedirs(OUT_DIR, exist_ok=True)

images = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if f.endswith((".jpg", ".png"))
])

for name in tqdm(images):

    img_path = os.path.join(IMAGE_DIR, name)
    img_bgr = cv2.imread(img_path)
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    H, W = img.shape[:2]

    base = os.path.splitext(name)[0]
    save_folder = os.path.join(OUT_DIR, base)
    os.makedirs(save_folder, exist_ok=True)

    # ---------- SAM masks ----------
    masks = mask_generator.generate(img)
    raw_masks = [m["segmentation"] for m in masks]

    filtered = []

    for m in raw_masks:
        area = m.sum()

        if area < MIN_AREA:
            continue

        if area > MAX_AREA_RATIO * H * W:
            continue

        if is_ground_like(m, H):
            continue

        filtered.append(m)

    filtered = remove_duplicate_masks(filtered)

    # ---------- SAVE MASKS ----------
    meta = []

    for i, m in enumerate(filtered):
        np.save(os.path.join(save_folder, f"mask_{i:03d}.npy"), m.astype(np.bool_))

        ys, xs = np.where(m)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

        meta.append({
            "id": i,
            "area": int(m.sum()),
            "bbox": bbox
        })

    with open(os.path.join(save_folder, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ---------- SAVE OVERLAY ----------
    save_overlay(img, filtered, os.path.join(save_folder, "overlay.jpg"))


print("\nDone.")
print("Masks + overlays saved to:", OUT_DIR)
