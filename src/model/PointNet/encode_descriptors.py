import argparse
import os
from os import listdir
from os.path import isfile, join
import glob
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
from get_neighbors import get_nbrs, get_weights, ensure_dir
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from get_neighbors import ensure_dir
from triplets_train import NeuralNetwork

def store_descriptors(d, dir, filepath_desc):
    path , file = os.path.split(filepath_desc)
    # pdb.set_trace()
    p = Path(dir)
    p.mkdir(exist_ok=True)
    file_name = os.path.join(dir,file)
    np.save(file_name, d)


def encode_descriptor(desc_files, obj_encoded_desc_dir, net):
    for f in desc_files:
        desc = np.load(f)
        enc_desc = net(torch.Tensor(desc).cuda())
        print("Storing encodings for fragment: ", f)
        # pdb.set_trace()
        store_descriptors(enc_desc.cpu().detach().numpy(), obj_encoded_desc_dir, f)

# names = ['brick', 'cube_6', 'cube_20',  'cake', 'gargoyle', 'head', 'sculpture', 'venus', 'cat', 'cat_low_res_0']
# for i in range(0,6):
#     if i == 2: continue
#     names.append(f'cat_seed_{i}')
# for i in range (1,15):
#     names.append(f'cube_20_seed_{i}')
# for i in range (1,10):
#     names.append(f'cylinder_20_seed_{i}')


KEYPOINT_PATH = "/home/sombit/dataset/Cube_8_seed_0/keypoints_2/"
DESC_PATH = "/home/sombit/dataset/Cube_8_seed_0/features/"
TRIPLET_PATH = "/home/sombit/dataset/Cube_8_seed_0/"
ENCODED_DESC = "/home/sombit/dataset/Cube_8_seed_0/"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
net = NeuralNetwork().to(device)
net.load_state_dict(torch.load('triplet_model_best.pth'))
net.eval()
print(net)

# for name in names[:2]:
#     OBJECT = name
#     # OBJECT = "cube_6"
#     USIP_TYPE = "_FN"
#     print("Processing: ", OBJECT+USIP_TYPE)
#     # kp_files= glob.glob(KEYPOINT_PATH+OBJECT+USIP_TYPE+"/*.npy")
#     desc_files = sorted(glob.glob(DESC_PATH+OBJECT+USIP_TYPE+"/*.npy"))
#     # print(kp_files)
#     obj_encoded_desc_dir = os.path.join(ENCODED_DESC, OBJECT + USIP_TYPE)
#     # print(desc_files)
#     ensure_dir(obj_encoded_desc_dir)
#     encode_descriptors(desc_files, obj_encoded_desc_dir, net)
PC_PATH = "/home/sombit/dataset/train1" 
object_type =os.listdir(PC_PATH)
if __name__ == '__main__':
    for object_t in object_type:
        print(object_t)
        object_t_dir = os.path.join(PC_PATH,object_t)
        objects = os.listdir(object_t_dir)
        for object_ in objects:
            obj_pc_dir = os.path.join(object_t_dir,object_)
            obj_triplet_dir = os.path.join(obj_pc_dir,"triplets_cls")
            obj_enc_dir = os.path.join(obj_pc_dir,"encoded")
            
            # pc_files= sorted(glob.glob(obj_pc_dir+"/*.npy"))
            # KEYPOINT_PATH = os.path.join(obj_pc_dir,'keypoints')
            # KEYPOINT_PATH = os.path.join(KEYPOINT_PATH,os.listdir(KEYPOINT_PATH)[0])
            # kp_files= sorted(glob.glob(os.path.join(KEYPOINT_PATH+"/*.npy")))

            DESC_PATH = os.path.join(obj_pc_dir,'features_cls')
            # KEYPOINT_PATH = os.path.join(KEYPOINT_PATH,os.listdir(KEYPOINT_PATH)[0])

            desc_files = sorted(glob.glob(DESC_PATH+"/*.npy"))
            encode_descriptor(desc_files, obj_enc_dir, net)
        #     break
        # break

        # KEYPOINT_PATH = os.path.join(KEYPOINT_PATH,os.listdir(KEYPOINT_PATH)[0])
        # triplet_path = os.path.join(obj_pc_dir,'triplets')
        # triplet_files.append(sorted(glob.glob(triplet_path+"/*.npz")))

# for name in names[:1]:
#     OBJECT = name
#     # OBJECT = "cube_6"
#     USIP_TYPE = "_1vN"
#     print("Processing: ", OBJECT+USIP_TYPE)
#     # kp_files= glob.glob(KEYPOINT_PATH+OBJECT+USIP_TYPE+"/*.npy")
#     desc_files = sorted(glob.glob(DESC_PATH+OBJECT+USIP_TYPE+"/*.npy"))
#     # print(kp_files)
#     obj_encoded_desc_dir = os.path.join(ENCODED_DESC, OBJECT + USIP_TYPE)
#     # print(desc_files)
#     ensure_dir(obj_encoded_desc_dir)
    
#     encode_descriptors(desc_files, obj_encoded_desc_dir, net)
