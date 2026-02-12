import os
import cv2
import numpy as np
import torch
import imageio

from segment_anything import sam_model_registry, SamPredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import matplotlib.pyplot as plt

# ---------------- CONFIG ---------------- #
image_dir = "/home/tejass/Downloads/Nu_scenes/nusc_ad_3d/images/"
save_dir = "/home/tejass/Downloads/Nu_scenes/nusc_ad_3d/grounded_sam_outputs/"
os.makedirs(save_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

sam_checkpoint = "sam_vit_l_0b3195.pth"
model_type = "vit_l"

TEXT_PROMPT = "car . truck . bus . pedestrian . cone . tree"

BOX_THRESHOLD = 0.35


# ---------------- LOAD SAM ---------------- #
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)
predictor = SamPredictor(sam)


# ---------------- LOAD GROUNDING DINO ---------------- #
model_id = "IDEA-Research/grounding-dino-tiny"

processor = AutoProcessor.from_pretrained(model_id)
dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
    model_id).to(device)


# ---------------- UTIL: overlay with per-class colors ---------------- #
def overlay_masks(image, masks, labels):
    overlay = image.copy()

    rng = np.random.RandomState(0)
    unique_labels = list(set(labels))

    color_map = {l: rng.randint(0, 255, 3) for l in unique_labels}

    for mask, lab in zip(masks, labels):
        color = color_map[lab]
        overlay[mask] = 0.5 * overlay[mask] + 0.5 * color

    return overlay.astype(np.uint8)


# ---------------- MAIN LOOP ---------------- #
image_files = sorted(
    [f for f in os.listdir(image_dir) if f.endswith((".jpg", ".png"))]
)

for idx, name in enumerate(image_files):
    print(f"[{idx+1}/{len(image_files)}] {name}")
    path = os.path.join(image_dir, name)

    image_bgr = cv2.imread(path)
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # -------- Grounding DINO detection --------
    inputs = processor(images=image, text=TEXT_PROMPT,
                       return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = dino_model(**inputs)  # gives logits, bboxes

    results = processor.post_process_grounded_object_detection(
        outputs,
        target_sizes=[image.shape[:2]]
    )[0]
    boxes = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    labels = results["labels"]  # list of strings, e.g., ["car", "truck"]

    keep = scores > BOX_THRESHOLD

    boxes = boxes[keep]
    scores = scores[keep]
    labels = [labels[i]
              for i in range(len(labels)) if keep[i]]  # filter strings manually

    if len(boxes) == 0:
        continue

    # -------- SAM segmentation --------
    predictor.set_image(image)
    masks_all = []
    mask_labels = []

    for box, lab, sc in zip(boxes, labels, scores):
        masks, mask_scores, _ = predictor.predict(
            box=box,
            multimask_output=True
        )

        best_mask = masks[np.argmax(mask_scores)]

        masks_all.append(best_mask)
        mask_labels.append(lab)

        print(f"  -> label={lab}, score={sc:.3f}")

    # -------- Save outputs --------
    base = os.path.splitext(name)[0]

    # Save individual masks
    # for i, (mask, lab) in enumerate(zip(masks_all, mask_labels)):
    #     mask_img = (mask.astype(np.uint8) * 255)

    #     np.save(os.path.join(save_dir, f"{base}_mask_{i}_label{lab}.npy"), mask_img)
    #     imageio.imwrite(os.path.join(save_dir, f"{base}_mask_{i}_label{lab}.png"), mask_img)

    # Save overlay
    overlay = overlay_masks(image, masks_all, mask_labels)
    imageio.imwrite(os.path.join(save_dir, f"{base}_overlay.png"), overlay)
    class_map = {'truck': 0, 'bus': 1, 'pedestrian': 2,
                 'car': 3, 'cone': 4, 'tree': 5}  # integer IDs

    # -------- Optional: semantic label map --------
    # each pixel stores class id
    semantic_map = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # Save EACH instance separately
    for i, (mask, lab) in enumerate(zip(masks_all, mask_labels)):
        mask_bool = mask.astype(bool)

        # Clean the label for file names
        lab_clean = lab.split()[0]  # take only the first word
        if lab_clean not in ["car", "truck", "bus", "pedestrian", "cone", "tree"]:
            continue  # skip anything unexpected

        # Save raw boolean mask
        np.save(os.path.join(save_dir, f"{base}_mask_{i}_label{lab_clean}.npy"), mask_bool)

        # Optional: save PNG for debugging
        imageio.imwrite(os.path.join(save_dir, f"{base}_mask_{i}_label{lab_clean}.png"), mask_bool.astype(np.uint8) * 255)


    # for mask, lab in zip(masks_all, mask_labels):
    #     # pick the first label if multiple
    #     lab_clean = lab.split()[0]
    #     if lab_clean in class_map and mask.sum() > 0:  # only assign if valid
    #         print(f"Assigning class {lab_clean} to {mask.sum()} pixels")
    #         semantic_map[mask] = class_map[lab_clean] + 1  # background=0
    # colors = np.array([
    #     [0, 0, 0],        # 0=background
    #     [255, 0, 0],      # 1=truck
    #     [0, 255, 0],      # 2=bus
    #     [0, 0, 255],      # 3=pedestrian
    #     [255, 255, 0],    # 4=car
    #     [255, 0, 255],    # 5=cone
    #     [0, 255, 255],    # 6=tree
    # ], dtype=np.uint8)

    # semantic_color = colors[semantic_map]
    # np.save(os.path.join(save_dir, f"{base}_semantic.npy"), semantic_map)

    # cv2.imwrite(os.path.join(
    #     save_dir, f"{base}_semantic_color.png"), semantic_color)

    # imageio.imwrite(os.path.join(save_dir, f"{base}_semantic.png"), semantic_map)


print("Done. Outputs saved to:", save_dir)
