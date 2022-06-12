from dis import dis
from operator import ne
from pickle import PROTO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial.distance import cdist
import time
import pdb
import glob
import os
from pathlib import Path
import torch



def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def positive_example(dist):
    min_dist = np.min(dist, axis=1)
    min_idx = np.argmin(dist, axis=1)
    return min_dist, min_idx

def negative_sample(dist,f_dist, min_dist, min_idx):
    dist_sub = dist - min_dist[:, None]
    dist_sub  = dist_sub - 0.04 ## threshold to consider examples outside  a ball of distance of 0.5
    dist_sub_1 = np.where(dist_sub>0, f_dist, np.inf)
    neg_idx = np.argmin(dist_sub_1, axis=1)
    # print(neg_idx.shape)
    neg_dist  = np.min(dist_sub_1, axis = 1)
    
    return neg_dist, neg_idx

def positive_abs(dist,positive_rad):
    # print("dist",dist,dist.shape)
    min_dist = np.min(dist, axis=1)
    # print("min_dist",min_dist,min_dist.shape)
    idx = np.where(min_dist<positive_rad)
    # print("idx",idx,idx[0].shape)
    min_idx = np.argmin(dist, axis=1)
    # print(min_idx)
    return min_dist[idx], min_idx[idx], idx

def negative_abs(dist,f_dist, min_dist, min_idx, idx,negative_rad):
    f_dist_filtered = f_dist
    dist_filtered = dist
    dist_sub = dist_filtered - negative_rad
    dist_sub_1 = np.where(dist_sub>0, f_dist_filtered, np.inf)

    neg_dist = np.min(dist_sub_1, axis=1)
    neg_idx = np.argmin(dist_sub_1, axis=1)
    return neg_dist, neg_idx

def store_triplet(anc, pos, neg, obj_triplet_dir, kp_file):
    path , f_temp = os.path.split(kp_file)
    file = f_temp[:-1] + 'z'

    p = Path(obj_triplet_dir)
    p.mkdir(exist_ok=True)
    file_name = os.path.join(obj_triplet_dir,file[:-4])
    np.savez(file_name, anchor=anc, positive=pos, negative = neg)

def generate_triplets(kp_files, desc_files, obj_triplet_dir,net,positive_rad,negative_rad):
    for i in range(len(kp_files)):

        ## file name for fragment processed
        frag = kp_files[i]
        # print(desc_files)
        desc = desc_files[i]
        # frag_inv = kp_files[i]
        # desc_inv = desc_files[i]
        
        ## file names for rest of the fragments
        other_frags = [f for f in kp_files if f != frag]
        other_d = [p for p in desc_files if p != desc]

        ## loading in the key points and descriptors for the fragment
        frag_kp = np.load(frag)
        frag_desc = np.load(desc)
        other_descs = np.zeros((1,128))
        other_kp = np.zeros((1,51))

        for p in other_d:
            desc_p = np.load(p)
            # print(desc_p.shape)
            other_descs = np.append(other_descs, desc_p, axis=0)

        for f in other_frags:
            kp_f = np.load(f)
            other_kp = np.append(other_kp, kp_f, axis =0 )

        other_kp = np.delete(other_kp, 0, axis = 0)
        other_descs = np.delete(other_descs, 0, axis = 0)

        ## compute distance between key points of frament i
        ## and the rest of the fragments
        dist = cdist(frag_kp[:,:3], other_kp[:,:3]) ## don't use sigma for distance calc
        f_dist = cdist(net(torch.Tensor(frag_desc[:,:]).cuda()).detach().cpu().numpy(), net(torch.Tensor(other_descs[:,:]).cuda()).detach().cpu().numpy())
        # if REL_DIST == 1:
        #     min_dist, min_idx = positive_example(dist)  ## find index for positive  pairs
        #     neg_dist, neg_idx = negative_sample(dist, f_dist, min_dist, min_idx)    ## find index for negative pairs
        # else:
        min_dist, min_idx, idx = positive_abs(dist,positive_rad)
        neg_dist, neg_idx = negative_abs(dist,f_dist, min_dist, min_idx, idx , negative_rad)

        ## anchor = desc
        ## positive descriptors
        ## negative descriptors

        anchor = frag_desc[idx]

        positive_desc = other_descs[min_idx]
        negative_desc = other_descs[neg_idx]
        # print(np.linalg.norm(frag_kp[idx][11,:3]-other_kp[min_idx][11,:3]))
        # print(np.linalg.norm(anchor[0,:128]-negative_desc[0,:128]) ,np.linalg.norm(frag_kp[idx][57,:3]-other_kp[neg_idx][57,:3]))
        # print("Storing triplets for fragment: ", i,anchor.shape, positive_desc.shape, negative_desc.shape)
        store_triplet(anchor, positive_desc, negative_desc, obj_triplet_dir, frag)
    # pdb.set_trace()
