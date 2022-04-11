import os
import numpy as np
from scipy.spatial.distance import cdist

"""
Loop over all fragments in input folder and find neighboring fragments
"""

DEFAULT_PATH = "Primitives_8/npy/Cube_8_seed_0"


def generate_neighbors(path, num_matches=100, dist_thresh=0.01):
    """
    Return neighboring matrix (symmetric) for all fragments in folder 'path'
    """
    print(f"Searching Neighbors in {path}")
    # load all .npy pointclouds
    files = [x for x in os.listdir(path) if x.endswith(".npy")]
    point_clouds = [np.load(os.path.join(path, f)) for f in files]
    
    return _neighbor_matrix(files, point_clouds, num_matches, dist_thresh)

def _neighbor_matrix(files, point_clouds, num_matches, dist_thresh):
    n_frag = len(files)
    neighbors = np.zeros((n_frag, n_frag))
    for i in range(n_frag):
        print(f"Progress: [{i*'#'+(n_frag-i)*' '}]", end='\r')
        for j in range(i+1, n_frag):
            neighbors[i,j] = _matches_count(point_clouds[i][:,:3], point_clouds[j][:,:3], num_matches, dist_thresh)
    neighbors += neighbors.T

    # print(neighbors)
    return neighbors

def _matches_count(pc1, pc2, num_matches, dist_thresh):
    matches = np.count_nonzero(cdist(pc1, pc2) < dist_thresh)
    return matches > num_matches


if __name__ == "__main__":
    generate_neighbors(DEFAULT_PATH)