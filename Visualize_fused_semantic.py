import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN

# ---------------- PATHS ----------------
fused_path = "/home/tejass/Downloads/Nu_scenes/nusc_ad_3d/colmap/dense/fused.ply"
semantic_path = "/home/tejass/Downloads/Nu_scenes/nusc_ad_3d/semantic_points.ply"

# ---------------- LOAD CLOUDS ----------------
fused = o3d.io.read_point_cloud(fused_path)
semantic = o3d.io.read_point_cloud(semantic_path)

print("Fused points:", len(fused.points))
print("Semantic points:", len(semantic.points))

# ---------------- CROP (apply same crop to both) ----------------
points = np.asarray(fused.points)
center = np.mean(points, axis=0)

distances = np.linalg.norm(points - center, axis=1)
max_radius = 100  # adjust as needed

inside_mask = distances < max_radius
fused = fused.select_by_index(np.where(inside_mask)[0])

# Crop semantic cloud using same center
sem_pts = np.asarray(semantic.points)
sem_dist = np.linalg.norm(sem_pts - center, axis=1)
semantic = semantic.select_by_index(np.where(sem_dist < max_radius)[0])

print("After crop:")
print("Fused:", len(fused.points))
print("Semantic:", len(semantic.points))

# ---------------- CLUSTER SEMANTIC POINTS ----------------
sem_points = np.asarray(semantic.points)
clustering = DBSCAN(eps=0.5, min_samples=10).fit(sem_points)
labels = clustering.labels_
num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"DBSCAN found {num_clusters} clusters")

# ---------------- FILTER PER-CLUSTER OUTLIERS ----------------
cluster_bboxes = []
max_cluster_radius = 0.3  # max distance from cluster centroid (meters)

for cluster_id in range(num_clusters):
    cluster_mask = labels == cluster_id
    cluster_pts = sem_points[cluster_mask]
    if len(cluster_pts) == 0:
        continue

    # compute centroid
    centroid = np.mean(cluster_pts, axis=0)
    distances = np.linalg.norm(cluster_pts - centroid, axis=1)
    cluster_pts = cluster_pts[distances < max_cluster_radius]

    if len(cluster_pts) == 0:
        continue

    # create Open3D point cloud for bounding box
    pc_cluster = o3d.geometry.PointCloud()
    pc_cluster.points = o3d.utility.Vector3dVector(cluster_pts)
    aabb = pc_cluster.get_axis_aligned_bounding_box()
    aabb.color = (1, 0, 0)  # red for visibility
    cluster_bboxes.append(aabb)

# ---------------- CLEAN DENSE CLOUD ONLY ----------------
fused, _ = fused.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

# ---------------- SEMANTIC POINTS NORMALS ----------------
semantic.estimate_normals()

# ---------------- VISUALIZATION ----------------
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Fused + Semantic 3D", width=1200, height=800)

vis.add_geometry(fused)
vis.add_geometry(semantic)

for box in cluster_bboxes:
    vis.add_geometry(box)

opt = vis.get_render_option()
opt.point_size = 5.0
opt.background_color = np.asarray([0.05, 0.05, 0.05])

vis.reset_view_point(True)
print("Dense cloud = grey, semantic points = colored")
vis.run()
vis.destroy_window()
