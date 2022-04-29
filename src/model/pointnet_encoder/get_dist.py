import argparse

  
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

np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)

import os
from os import listdir
from os.path import isfile, join
import glob
from pyexpat import features
from webbrowser import get
from data_utils.S3DISDataLoader import ScannetDatasetWholeScene
from data_utils.indoor3d_util import g_label2color
import torch
import logging
from pathlib import Path
import sys
import importlib
from tqdm import tqdm
import provider
import numpy as np
import pdb
from torchsummary import summary
from get_neighbors import *
from find_distances import *
# %matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PC_PATH = "/home/sombit/object_new/" 
from tqdm import tqdm
# triplet_files = []
# for name in names[:2]:
#     OBJECT = name
#     # OBJECT = "cube_6"
#     USIP_TYPE = "_FN"
#     print("Processing: ", OBJECT)
#     kp_files= sorted(glob.glob(KEYPOINT_PATH+OBJECT+USIP_TYPE+"/*.npy"))
#     desc_files = sorted(glob.glob(DESC_PATH+OBJECT+USIP_TYPE+"/*.npy"))
#     triplet_files.append(sorted(glob.glob(TRIPLET_PATH+OBJECT+USIP_TYPE+"/*.npz")))
# for name in names[:2]:
#     OBJECT = name
#     # OBJECT = "cube_6"
#     USIP_TYPE = "_1vN"
#     print("Processing: ", OBJECT)
#     kp_files= sorted(glob.glob(KEYPOINT_PATH+OBJECT+USIP_TYPE+"/*.npy"))
#     desc_files = sorted(glob.glob(DESC_PATH+OBJECT+USIP_TYPE+"/*.npy"))
#     triplet_files.append(sorted(glob.glob(TRIPLET_PATH+OBJECT+USIP_TYPE+"/*.npz")))
object_type =os.listdir(PC_PATH)


def get_matr(kp,desc_files):
    pos=0
    neg = 0
    off =0
    for f_kp in tqdm(  range(len(kp_files))):
        print("Processing: ", f_kp)
        kp = np.load(kp_files[f_kp])[:,:3]
        desc = np.load(desc_files[f_kp]) 
        test = np.linalg.norm(desc,axis=1)
        print(np.min(test),np.max(test),np.mean(test))
        # for i in range(kp.shape[0]):
        #     dist.append(np.linalg.norm(kp[0] - kp[i]))
        #     feat.append(np.linalg.norm(desc[0] - desc[i]))
        for ind in tqdm(range(kp.shape[0])):
            dist = []
            feat = []
            for f_nnkp in range(len(kp_files)):
                # print("             Processing: ", f_nnkp)
                if(f_nnkp !=f_kp):
                    kpp_n = np.load(kp_files[f_nnkp])[:,:3]
                    desc_n = np.load(desc_files[f_nnkp])
                    for i in range(kpp_n.shape[0]):
                        dist.append(np.linalg.norm(kp[ind] - kpp_n[i]))
                        feat.append(np.linalg.norm(desc[ind] - desc_n[i]))
            # if(np.min(dist)<0.1):
            #     get_distances(np.array(dist),np.array(feat))   
            if( dist[np.argmin(feat)]<0.1):
                pos+=1
            else:
                if(np.min(feat)<0.1 and dist[np.argmin(feat)]>0.1):
                    neg+=1
                else:
                    off+=1
        print(pos,neg,off)   
        # ax = plt.axes()
    #     for i in range(kp.shape[0]):
    #         dist.append(np.linalg.norm(kp[0] - kp[i]))
    #         feat.append(np.linalg.norm(desc[0] - desc[i]))
    # fig = plt.figure()
    # plt.plot(np.array(dist),np.array(feat),'bo')
    # plt.show()

    # for f_kp in desc_files:            
         
    
for object_t in object_type:
    print(object_t)
    object_t_dir = os.path.join(PC_PATH,object_t)
    objects = os.listdir(object_t_dir)
    # kp_files_total = []
    # desc_files_total = []
    for object_ in objects:
        obj_pc_dir = os.path.join(object_t_dir,object_)
        obj_triplet_dir = os.path.join(obj_pc_dir,"triplets_cls")
        pc_files= sorted(glob.glob(obj_pc_dir+"/*.npy"))
        KEYPOINT_PATH = os.path.join(obj_pc_dir,'keypoints')
        # KEYPOINT_PATH = os.path.join(KEYPOINT_PATH,os.listdir(KEYPOINT_PATH)[0])
        # kp_files= sorted(glob.glob(os.path.join(KEYPOINT_PATH+"/*.npy")))
        kp_files= sorted(glob.glob(os.path.join(KEYPOINT_PATH+"/*.npy")))
        print(kp_files)
        # kp_files_total.append(kp_files)

        DESC_PATH = os.path.join(obj_pc_dir,'encoded_cls')
        desc_files = sorted(glob.glob(DESC_PATH+"/*.npy"))
        # desc_files_total.append(desc_files_total)
        # get_matr(kp_files,desc_files)
        break
    break

i = 0
j =1
n_points  = 512
colors1 = ['blue' for _ in range(n_points)]
colors2 = ['blue' for _ in range(n_points)]
x_lines = []
y_lines = []
z_lines = []
c = 2
kp_i = np.load(kp_files[i])[:,:3]
kp_j = np.load(kp_files[j])[:,:3]

pc_i = np.load(pc_files[i])
pc_j = np.load(pc_files[j])

ds_i = np.load(desc_files[i])
ds_j = np.load(desc_files[j])
fig = go.Figure()
# for k in range(len(kp_files)):
#     pc_j = np.load(pc_files[k])
#     fig.add_trace(
#     go.Scatter3d(
#         x = pc_j[:, 0],
#         y = pc_j[:, 1],
#         z = pc_j[:, 2],
#         mode='markers',
#         marker=dict(
#             size=1,
#         )
#     )
# )
pos = 0
neg =0
for i in range(n_points):
    d = []
    dist = []
    for j in range(n_points):
        dist.append(np.linalg.norm(kp_i[i] - kp_j[j]))
    if(np.min(dist)<0.001):
        for j in range(n_points):
            d.append(np.linalg.norm(ds_i[i] - ds_j[j]))
        if(np.argmin(dist) == np.argmin(d)):
            pos+=1
        else:
            neg+=1
        j = np.argmin(np.array(d))
        
        colors1[i] = 'green'
        colors2[j] = 'green'

        x_lines.append(kp_i[i][0])
        x_lines.append(kp_j[j][0]+c)
        x_lines.append(None)

        y_lines.append(kp_i[i][1])
        y_lines.append(kp_j[j][1])
        y_lines.append(None)

        z_lines.append(kp_i[i][2])
        z_lines.append(kp_j[j][2])
        z_lines.append(None)

print(pos,neg)

fig.add_trace(
    go.Scatter3d(
        x = pc_i[:, 0],
        y = pc_i[:, 1],
        z = pc_i[:, 2],
        mode='markers',
        marker=dict(
            size=1,
        )
    )
)
fig.add_trace(
    go.Scatter3d(
        x = kp_i[:, 0],
        y = kp_i[:, 1],
        z = kp_i[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color=colors1
        )
    )
)
kp_j[:, 0] += c
pc_j[:, 0] += c
fig.add_trace(
    go.Scatter3d(
        x = pc_j[:, 0],
        y = pc_j[:, 1],
        z = pc_j[:, 2],
        mode='markers',
        marker=dict(
            size=1,
        )
    )
)
fig.add_trace(
    go.Scatter3d(
        x = kp_j[:, 0],
        y = kp_j[:, 1],
        z = kp_j[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color=colors2
        )
    )
)

fig.add_trace(
    go.Scatter3d(
        x = x_lines,
        y = y_lines,
        z = z_lines,
        mode='lines',
    )
)

fig.show()

	
#         break
#     break

        # KEYPOINT_PATH = os.path.join(KEYPOINT_PATH,os.listdir(KEYPOINT_PATH)[0])
        # triplet_path = os.path.join(obj_pc_dir,'triplets')
        # triplet_files.append(sorted(glob.glob(triplet_path+"/*.npz")))
        # print(kp_files)
            # generate_triplets(kp_files, desc_files, obj_triplet_dir)
            # print(obj_f_dir)