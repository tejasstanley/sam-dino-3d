import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN

semantic_path = "/home/tejass/Downloads/Nu_scenes/nusc_ad_3d/semantic_points.ply"
semantic = o3d.io.read_point_cloud(semantic_path)

# Voxel downsample to reduce number of points
voxel_size = 0.1  # meters, adjust based on scene
semantic = semantic.voxel_down_sample(voxel_size)
points = np.asarray(semantic.points)
colors = np.asarray(semantic.colors)

print(f"Downsampled points: {len(points)}")

db = DBSCAN(eps=0.5, min_samples=10).fit(points)
labels = db.labels_
mask = labels != -1

clustered_points = points[mask]
clustered_colors = colors[mask]

clustered_cloud = o3d.geometry.PointCloud()
clustered_cloud.points = o3d.utility.Vector3dVector(clustered_points)
clustered_cloud.colors = o3d.utility.Vector3dVector(clustered_colors)

o3d.visualization.draw_geometries([clustered_cloud], window_name="DBSCAN Clustered Semantic Points")
