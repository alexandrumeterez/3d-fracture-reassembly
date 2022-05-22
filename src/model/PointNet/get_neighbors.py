import os
import numpy as np
import pdb
from sklearn.neighbors import KDTree

def get_weights(dist):
    w = 1 / dist
    w_sum = np.sum(w, axis=1)
    rel_w = w / w_sum[ : , None]
    return rel_w

def get_nbrs(point_cloud, key_points):
    kdt = KDTree(point_cloud[:,:3], leaf_size=50, metric='euclidean')
    dist , idx = kdt.query(key_points, k=1, return_distance=True)
    return dist, idx

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)