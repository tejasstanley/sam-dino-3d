from nuscenes.nuscenes import NuScenes
import os
import shutil

# Change this path
NUSC_ROOT = "/home/tejass/Downloads/Nu_scenes"
OUT_DIR = "nusc_ad_3d/images"

nusc = NuScenes(version="v1.0-trainval", dataroot=NUSC_ROOT, verbose=True)

# Pick a scene index (try 0 first)
scene = nusc.scene[0]
print("Scene name:", scene["name"])

# Get first sample in scene
sample_token = scene["first_sample_token"]
print ("First sample token:", sample_token)

os.makedirs(OUT_DIR, exist_ok=True)

count = 0
MAX_FRAMES = 100  # change to 50 if you want faster runs

while sample_token and count < MAX_FRAMES:
    sample = nusc.get("sample", sample_token)

    cam_token = sample["data"]["CAM_FRONT"]

    src = nusc.get_sample_data_path(cam_token)
    dst = os.path.join(OUT_DIR, f"{count:06d}.jpg")

    shutil.copy(src, dst)

    sample_token = sample["next"]
    count += 1

print("Extracted frames:", count)
