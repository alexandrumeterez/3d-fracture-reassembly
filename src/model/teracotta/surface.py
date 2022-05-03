from turtle import color
import numpy as np
from scipy import spatial
import os
import argparse
import time
import scipy
import plotly
import plotly.graph_objects as go
from tqdm import tqdm
from plotly.subplots import make_subplots 
from scipy.optimize import curve_fit
from multiprocessing import Pool
import pandas as pd
import pyvista as pv
np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)
def get_parent(dict_map,i):
    if(dict_map[i]==i):
        return i
    else:
        return get_parent(dict_map,dict_map[i]) 
if __name__ == '__main__':
    shard = 2
    fragment_path = '/home/sombit/object_test/Cone_16_seed_1/Cone_shard.001.npy'
    fragment = np.load(fragment_path)
    
    point_cloud = fragment[:, :3]
    # cloud = pv.PolyData(point_cloud)
    # cloud.plot()

    # volume = cloud.delaunay_3d(alpha=.1)
    # shell = volume.extract_geometry()
    # shell.plot()
    # normals = fragment[:, 3:].T
    tree = spatial.KDTree(point_cloud)
    # params
    k = 150
    lbd = 2
    dists, nbhd = tree.query(x=point_cloud, k=k, workers=-1)
    labels = []
    colors = []

    for i in range(point_cloud.shape[0]):
        # get neighboring points
        closest_points = nbhd[i]
        C_i = np.mean(point_cloud[closest_points], axis=0)
        Z_i = dists[i][1]
        
        score = np.linalg.norm(C_i - point_cloud[i])
        # if(score > lbd * Z_i):
        #     colors.append("red")
        # else:
        #     colors.append("blue")
    
        colors.append(score)
        labels.append(score > lbd * Z_i)
    print(np.min(colors), np.max(colors), np.mean(colors))
    dist,neighbors = tree.query(x=point_cloud, k=60,distance_upper_bound=0.1, workers=-1)
    dict_new = {}
    for i in range(point_cloud.shape[0]):
        dict_new[i] = {i}
    for i in range(point_cloud.shape[0]):
        for j in neighbors[i]:
            if(np.abs(colors[i] - colors[j]) < 0.06):
                dict_new[j] = i
                # dict_new[i].union( dict_new[j])
                # print(dict_new[i],i,j)
                # del dict_new[j]   
    surf = {}
    for i in range(point_cloud.shape[0]):
        k = get_parent(dict_new,i)
        if(k in surf.keys()):
            surf[k].append(i)
        else:
            surf[k] = [i]
    print(len(surf.keys()),point_cloud.shape[0])
    labels = np.asarray(labels)
    fig = go.Figure()
    for k in surf.keys():
        # for j in surf[k]:
  
        fig.add_trace(
            go.Scatter3d(
                x = np.array(point_cloud[np.array(surf[k]), 0]),
                y = np.array(point_cloud[np.array(surf[k]), 1]),
                z = np.array(point_cloud[np.array(surf[k]), 2]),
                mode='markers',
                marker=dict(
                    size=3,
                    color=k
                )
            )
        )
    fig.show()