import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import euclidean

# Given set of points
pts = np.array([[-0.85098575, 0.95852071], [-0.80569344, 1.45690896], [-1.07024601, 0.92975891],
                [-0.52623616, 1.23023042], [-1.14084232, 1.16276801], [-1.13902531, 0.86028107],
                [-0.92741132, 0.42601593], [-1.51747535, 0.83131374], [-1.30384934, 1.0942742],
                [-1.27240722, 0.57630889], [-0.56030537, 0.93226711], [-0.97974154, 0.57257554],
                [-1.16331482, 1.03327678], [-1.34529807, 1.11270941], [-1.18019161, 0.91249188],
                [-1.18051198, 1.55568346], [-1.00404917, 0.68268672], [-0.75323653, 0.63374691],
                [-0.93734092, 0.41209896], [-1.39845581, 1.05905837], [-0.77846003, 1.05141048],
                [-1.03469448, 0.90966889], [-1.4435566, 0.78404674], [-1.13819163, 1.31713667],
                [-0.89691451, 0.47108795], [1.09722519, -0.19254114], [0.7969234, 0.30583814],
                [1.30929986, 0.46564006], [0.74823474, -0.15460619], [1.09937903, 0.48777256],
                [0.85624773, -0.09282949], [0.66809951, -0.59810331], [1.24375775, 0.67812001],
                [0.97839696, 0.50176645], [1.10849081, -0.32255988], [1.10841868, 0.76901828],
                [0.98925219, 0.78232183], [0.21407647, 0.41095125], [1.02611412, -0.14950368],
                [1.02752823, -0.99378446], [-1.10983594, 0.10713377], [-0.26105298, -0.15548107],
                [-1.4042468, -0.15052711], [-0.54229894, 0.09862533], [-1.2648801, 0.15398023],
                [-0.95146123, 0.2905935], [-1.35102655, -0.09829864], [-1.19605408, -0.43905448],
                [-0.85193986, 0.07831658], [-0.99744327, -0.07037614]])
plt.scatter(pts[:, 0], pts[:, 1], c='black', marker='o', label='Data Points')
plt.title('Step 1: Scatter Plot of Data Points')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()

knn_model = NearestNeighbors(n_neighbors=5)
knn_model.fit(pts)
densities = knn_model.kneighbors(pts, return_distance=False).sum(axis=1)

sorted_indices = np.argsort(densities)[::-1]
threshold = 0.75
seed_points = []
for i in sorted_indices:
    if not seed_points or euclidean(pts[i], seed_points[-1]) > threshold:
        seed_points.append(pts[i])
    if len(seed_points) == 3:
        break

clusters = [[] for _ in range(3)]
for point in pts:
    distances = [euclidean(point, seed) for seed in seed_points]
    closest_seed = np.argmin(distances)
    clusters[closest_seed].append(point)

colors = ['red', 'green', 'blue']
for i, cluster in enumerate(clusters):
    cluster = np.array(cluster)
    plt.scatter(cluster[:, 0], cluster[:, 1], c=colors[i], marker='o', label=f'Cluster {i+1}')
plt.scatter(np.array(seed_points)[:, 0], np.array(seed_points)[:, 1], c='black', marker='x', label='Seed Points')
plt.title('Step 5: Clusters with Seed Points')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.show()
