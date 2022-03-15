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
    kdt = KDTree(point_cloud, leaf_size=50, metric='euclidean')
    dist , idx = kdt.query(key_points, k=10, return_distance=True)
    return dist, idx

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

KEYPOINT_PATH = "./data/keypoints/keypoints_2/keypoints_2/"
PC_PATH="./data/keypoints/real_data_npy/npy/"
OBJECT="brick"
USIP_TYPE = "_1vN/"
FRG="0"
PC_file="/brick_part_0.npy"

pc = np.load(PC_PATH+OBJECT+PC_file)
kp = np.load(KEYPOINT_PATH+OBJECT+USIP_TYPE+FRG + '.npy')

sigma = kp[:,3]
kp = kp[:,:3]
pc_coord = pc[:,:3]
# tup =get_nbrs(pc_coord, kp)
# print(tup)
## no key point directly lies in the point cloud
# for k in kp:
#     if k in pc_coord:
#         print(k)


kdt = KDTree(pc_coord, leaf_size=30, metric='euclidean')
dist, idx = kdt.query(kp, k=10, return_distance=True)
# pdb.set_trace()
