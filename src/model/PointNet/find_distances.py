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

sigma=0.001

n = 0
m = 0
p = 0
REL_DIST = 0
# KEYPOINT_PATH = "./data/keypoints/keypoints_2/keypoints_2/"
KEYPOINT_PATH = "./data/keypoints/keypoints_4/"

DESC_PATH = "./data/keypoints/features/"
names=[]
names.extend(['brick','cube_6','cube_20', 'cake', 'gargoyle', 'head', 'sculpture', 'venus', 'cat', 'cat_low_res_0'])
for i in range(0,6):
    if i == 2: continue
    names.append(f'cat_seed_{i}')
for i in range (1,15):
    names.append(f'cube_20_seed_{i}')
for i in range (1,10):
    names.append(f'cylinder_20_seed_{i}')

# print(names)
#

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def positive_example(dist):
    min_dist = np.min(dist, axis=1)
    min_idx = np.argmin(dist, axis=1)
    return min_dist, min_idx

def negative_sample(dist, min_dist, min_idx):
    dist_sub = dist - min_dist[:, None]
    dist_sub  = dist_sub - 0.5 ## threshold to consider examples outside  a ball of distance of 0.5
    dist_sub_1 = np.where(dist_sub>0, dist_sub, np.inf)
    neg_dist  = np.min(dist_sub_1, axis = 1)
    neg_idx = np.argmin(dist_sub_1, axis=1)
    return neg_dist, neg_idx

def positive_abs(dist):
    min_dist = np.min(dist, axis=1)
    idx = np.where(min_dist<0.5)
    min_idx = np.argmin(dist, axis=1)
    return min_dist[idx], min_idx[idx], idx

def negative_abs(dist, min_dist, min_idx, idx):
    dist_filtered = dist[idx]
    dist_sub = dist_filtered - 0.5
    dist_sub_1 = np.where(dist_sub>0, dist_sub, np.inf)
    neg_dist = np.min(dist_sub_1, axis=1)
    neg_idx = np.argmin(dist_sub_1, axis=1)
    return neg_dist, neg_idx

def store_triplet(anc, pos, neg, obj_triplet_dir, kp_file):
    path , f_temp = os.path.split(kp_file)
    file = f_temp[:-1] + 'z'
    file_name = os.path.join(obj_triplet_dir,file)
    np.savez(file_name, anchor=anc, positive=pos, negative = neg)

def generate_triplets(kp_files, desc_files, obj_triplet_dir):
    for i in range(len(kp_files)):

        ## file name for fragment processed
        frag = kp_files[i]
        desc = desc_files[i]
        ## file names for rest of the fragments
        other_frags = [f for f in kp_files if f != frag]
        other_d = [p for p in desc_files if p != desc]

        ## loading in the key points and descriptors for the fragment
        frag_kp = np.load(frag)
        frag_desc = np.load(desc)

        other_descs = np.zeros((1,128))
        other_kp = np.zeros((1,4))

        for p in other_d:
            desc_p = np.load(p)
            other_descs = np.append(other_descs, desc_p, axis=0)

        for f in other_frags:
            kp_f = np.load(f)
            other_kp = np.append(other_kp, kp_f, axis =0 )

        other_kp = np.delete(other_kp, 0, axis = 0)
        other_descs = np.delete(other_descs, 0, axis = 0)

        ## compute distance between key points of frament i
        ## and the rest of the fragments
        dist = cdist(frag_kp[:,:3], other_kp[:,:3]) ## don't use sigma for distance calc

        if REL_DIST == 1:
            min_dist, min_idx = positive_example(dist)  ## find index for positive  pairs
            neg_dist, neg_idx = negative_sample(dist, min_dist, min_idx)    ## find index for negative pairs
        else:
            min_dist, min_idx, idx = positive_abs(dist)
            neg_dist, neg_idx = negative_abs(dist, min_dist, min_idx, idx)

        ## anchor = desc
        ## positive descriptors
        ## negative descriptors
        anchor = frag_desc[idx]
        positive_desc = other_descs[min_idx]
        negative_desc = other_descs[neg_idx]
        print("Storing triplets for fragment: ", i)
        store_triplet(anchor, positive_desc, negative_desc, obj_triplet_dir, frag)
    # pdb.set_trace()



# name = names[0] #OBJECT="brick"
for name in names[:1]:
    OBJECT = name
    # OBJECT = "cube_6"
    # USIP_TYPE = "_FN"
    USIP_TYPE = "_1vN"

    print("Processing: ", OBJECT)
    kp_files= sorted(glob.glob(KEYPOINT_PATH+OBJECT+USIP_TYPE+"/*.npy"))
    desc_files = sorted(glob.glob(DESC_PATH+OBJECT+USIP_TYPE+"/*.npy"))
    # print(kp_files)
    triplet_dir = "./data/keypoints/triplets/"
    obj_triplet_dir = os.path.join(triplet_dir, OBJECT + USIP_TYPE)
    ensure_dir(obj_triplet_dir)

    generate_triplets(kp_files, desc_files, obj_triplet_dir)



    # fig = pl.hist(a,normed=0)
    # pl.title('Mean')
    # pl.xlabel("value")
    # pl.ylabel("Frequency")
    # pl.savefig("abc.png")







# for name in names:
#     print(f'\nProcessing {name}...')
#     fragment=[]
#     keypoints=[]
#
#     # Load keypoints of each fragment
#     for part in range(1000):
#         try:
#             fragment.append(np.load(f'KEYPOINT_PATH+{name}_FN/{part}.npy'))
#         except:
#             print(f'{part} parts loaded.\n')
#             break
#
#         print(fragment[part].shape)
#
#         if part==999:
#             print('WARNING: part limit reached, only loaded the first 1000 parts.')
#             part += 1
#
#     # scatter plot of pointcloud and keypoints
#     # fig = plt.figure()
#     # ax = fig.add_subplot(projection='3d')
#     # for i in range(part):
#     #     ax.scatter(fragment[i][:,0], fragment[i][:,2], fragment[i][:,1], color=cm.Set1(i/part) , s=1)#, fragment[:,3], fragment[:,4], fragment[:,5])
#     # plt.show()
#
#     distance=[]
#     index=[]
#
#     for i in range(part):
#         this_dist=np.zeros((fragment[i].shape[0],1))
#         fr_idx=[]
#         kp_idx=[]
#         for j in range(part):
#             if i == j: continue
#             # get distance matrix from all keypoints of current part (i) to all other keypoints (part j)
#             this_dist=np.concatenate((this_dist, cdist(fragment[i][:,:3],fragment[j][:,:3])),axis=1)
#             # create index, consisting of part number and keypoint index
#             fr_idx.extend([j]*fragment[j].shape[0])
#             kp_idx.extend(list(range(fragment[j].shape[0])))
#         # delete first column of distance matrix (column of zreos was needed for concatenation to work )
#         this_dist=np.delete(this_dist, 0, axis=1)
#         # create real index (part, keypoint), sort it by distance
#         this_idx=np.transpose(np.array([fr_idx,kp_idx]))
#         this_idx=this_idx[np.argsort(this_dist)]
#         # sort distance matrix by distance
#         this_dist=np.sort(this_dist)
#
#         distance.append(this_dist)
#         index.append(this_idx)
#
#
# pdb.set_trace()
# # index[i]: index matrix of fragment i; index[i][j,k]: gives the fragment number and keypoint index closest to keypoint j on fragment i, ordered by distance
# # distance[i]: distance matrix of fragment i; distance[i][j,k] gives the distance from keypoint j of fragment i to keypoint, given by index[i][j,k]
