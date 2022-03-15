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
    file_name = os.path.join(dir,file)
    # pdb.set_trace()
    np.save(file_name, d)


def encode_descriptors(desc_files, obj_encoded_desc_dir, net):
    for f in desc_files:
        desc = np.load(f)
        enc_desc = net(torch.Tensor(desc))
        print("Storing encodings for fragment: ", f)
        # pdb.set_trace()
        store_descriptors(enc_desc.cpu().detach().numpy(), obj_encoded_desc_dir, f)

names = ['brick', 'cube_6', 'cube_20',  'cake', 'gargoyle', 'head', 'sculpture', 'venus', 'cat', 'cat_low_res_0']
for i in range(0,6):
    if i == 2: continue
    names.append(f'cat_seed_{i}')
for i in range (1,15):
    names.append(f'cube_20_seed_{i}')
for i in range (1,10):
    names.append(f'cylinder_20_seed_{i}')


KEYPOINT_PATH = "./data/keypoints/keypoints_2/keypoints_2/"
DESC_PATH = "./data/keypoints/features/"
TRIPLET_PATH = "./data/keypoints/triplets/"
ENCODED_DESC = "./data/keypoints/encoded_desc/"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
net = NeuralNetwork().to(device)
net.eval()
print(net)

for name in names[:2]:
    OBJECT = name
    # OBJECT = "cube_6"
    USIP_TYPE = "_FN"
    print("Processing: ", OBJECT+USIP_TYPE)
    # kp_files= glob.glob(KEYPOINT_PATH+OBJECT+USIP_TYPE+"/*.npy")
    desc_files = sorted(glob.glob(DESC_PATH+OBJECT+USIP_TYPE+"/*.npy"))
    # print(kp_files)
    obj_encoded_desc_dir = os.path.join(ENCODED_DESC, OBJECT + USIP_TYPE)
    # print(desc_files)
    ensure_dir(obj_encoded_desc_dir)
    encode_descriptors(desc_files, obj_encoded_desc_dir, net)

for name in names[:1]:
    OBJECT = name
    # OBJECT = "cube_6"
    USIP_TYPE = "_1vN"
    print("Processing: ", OBJECT+USIP_TYPE)
    # kp_files= glob.glob(KEYPOINT_PATH+OBJECT+USIP_TYPE+"/*.npy")
    desc_files = sorted(glob.glob(DESC_PATH+OBJECT+USIP_TYPE+"/*.npy"))
    # print(kp_files)
    obj_encoded_desc_dir = os.path.join(ENCODED_DESC, OBJECT + USIP_TYPE)
    # print(desc_files)
    ensure_dir(obj_encoded_desc_dir)
    encode_descriptors(desc_files, obj_encoded_desc_dir, net)
