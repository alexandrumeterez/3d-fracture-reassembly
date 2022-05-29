import argparse
import time
import open3d as o3d
import scipy
import plotly
import plotly.graph_objects as go
import numpy as np
import os
from tqdm import tqdm
from plotly.subplots import make_subplots 
from scipy.optimize import curve_fit
from multiprocessing import Pool
import pandas as pd
np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--voxel-size", type=float, required=True)
    args = parser.parse_args()

    # Load fragment
    fragments = os.listdir(args.dataset_dir)
    subsampled_dir = os.path.join(args.dataset_dir, "subsampled") 
    if os.path.exists(subsampled_dir):
        import shutil
        shutil.rmtree(subsampled_dir)
        os.mkdir(subsampled_dir)
    else:
        os.mkdir(subsampled_dir)
    for fragment in fragments:
        if not fragment.endswith(".npy"):
            continue
        fragment_path = os.path.join(args.dataset_dir, fragment)
        output_path = os.path.join(args.dataset_dir, "subsampled", fragment) 
        point_cloud = np.load(fragment_path)
        before_size = point_cloud.shape[0]
        pcd = o3d.geometry.PointCloud() 
        pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
        pcd.normals = o3d.utility.Vector3dVector(point_cloud[:, 3:])

        pcd = pcd.voxel_down_sample(voxel_size=args.voxel_size)
        
        n_points = np.array(pcd.points).shape[0]
        print(f"From {before_size} to {n_points}")
        output_pcd = np.zeros((n_points, 6))
        output_pcd[:, :3] = np.array(pcd.points)
        output_pcd[:, 3:] = np.array(pcd.normals)
        
        np.save(output_path, output_pcd)
