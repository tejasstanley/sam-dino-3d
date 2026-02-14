Current Pipeline:
1. Run COLMAPS to get the sparse point cloud and catmera parameters.
2. Run DINO and SAM to get the masks for the objects in the images.
3. Lift the 2D masks to 3D using the camera parameters and the sparse point cloud.
4. Use the lifted masks to reconstruct the 3D shape of the objects.
5. Cluster the points to get the final 3D model of the objects.

TODOS:
1. Currently SAM and DINO give the optimmum masks based on queries. 
2. Lifting from 2D to 3D is not good enough. Sparse COLMAPS gives very few points. Dense COLMAP projection giving unnecessary points. 
3. Need a better Reconstructructed Data. NuScenes not good for static objects.
4. Proceeded with sparse COLMAPS for now. Clustered the points, but clusters getting 