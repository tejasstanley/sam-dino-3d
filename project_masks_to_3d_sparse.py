# project_masks_to_3d.py

import os
import numpy as np
import pycolmap
from collections import defaultdict

# ---------------- CONFIG ---------------- #

colmap_model = "/home/tejass/Downloads/Nu_scenes/nusc_ad_3d/colmap/sparse/0"
mask_dir = "/home/tejass/Downloads/Nu_scenes/nusc_ad_3d/grounded_sam_outputs/"
output_ply = "/home/tejass/Downloads/Nu_scenes/nusc_ad_3d/semantic_points.ply"

class_map = {
    1: "truck",
    2: "bus",
    3: "pedestrian",
    4: "car",
    5: "cone",
    6: "tree"
}

MIN_POINTS = 30   # ignore tiny noisy clusters

# ---------------- LOAD COLMAP ---------------- #

print("Loading COLMAP model...")
rec = pycolmap.Reconstruction(colmap_model)

print("Images:", len(rec.images))
print("3D points:", len(rec.points3D))

# ---------------- STORAGE ---------------- #

class_points = defaultdict(list)

# ---------------- MAIN LOOP ---------------- #
# Iterate over COLMAP images
for img_id, img in rec.images.items():
    name = os.path.splitext(img.name)[0]

    # Find all mask files for this image
    mask_files = [f for f in os.listdir(mask_dir) if f.startswith(name) and f.endswith(".npy")]

    if len(mask_files) == 0:
        continue

    print("Processing:", img.name)

    # Iterate over all 2D points with 3D correspondence
    for p in img.points2D:
        if not p.has_point3D():
            continue
        pid = p.point3D_id
        u, v = int(p.xy[0]), int(p.xy[1])

        # Find which mask this pixel belongs to
        for mask_file in mask_files:
            mask_path = os.path.join(mask_dir, mask_file)
            mask = np.load(mask_path)  # boolean mask
            if mask[v, u]:  # pixel inside mask
                # extract class from filename
                cls = mask_file.split("label")[-1].split(".")[0]
                xyz = rec.points3D[pid].xyz
                class_points[cls].append(xyz)
                break  # stop after first mask match
            
            
            
with open(output_ply, 'w') as f:
    # Write header
    total_points = sum(len(v) for v in class_points.values())
    f.write("ply\nformat ascii 1.0\n") # ASCII format for easy debugging, can switch to binary for efficiency
    f.write(f"element vertex {total_points}\n") # total number of points
    f.write("property float x\nproperty float y\nproperty float z\n") # 3D coordinates
    f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n") # RGB color for visualization
    f.write("end_header\n") # end of header, now write point data

    # Color map
    color_map = {
        "truck": (255, 0, 0),
        "bus": (0, 255, 0),
        "pedestrian": (0, 0, 255),
        "car": (255, 255, 0),
        "cone": (255, 0, 255),
        "tree": (0, 255, 255),
    }

    # Write points
    for cls, pts in class_points.items():
        r, g, b = color_map.get(cls, (255, 255, 255))
        for xyz in pts:
            f.write(f"{xyz[0]} {xyz[1]} {xyz[2]} {r} {g} {b}\n")
print("3D points saved to:", output_ply)