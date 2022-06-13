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



device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
net = NeuralNetwork().to(device)
net.load_state_dict(torch.load('triplet_model_best.pth'))
net.eval()

PC_PATH = "/home/sombit/dataset/test" 
objects = os.listdir(PC_PATH)
for object_ in objects:
        obj_pc_dir = os.path.join(PC_PATH,object_)
        # obj_triplet_dir = os.path.join(obj_pc_dir,"triplets")
        pc_files= sorted(glob.glob(obj_pc_dir+"/*.npy"))
        # print(obj_pc_dir)
        KEYPOINT_PATH = os.path.join(obj_pc_dir,'keypoints')
        # KEYPOINT_PATH = os.path.join(KEYPOINT_PATH,os.listdir(KEYPOINT_PATH)[0])
        # kp_files= sorted(glob.glob(os.path.join(KEYPOINT_PATH+"/*.npy")))
        kp_files= sorted(glob.glob(os.path.join(KEYPOINT_PATH+"/*.npy")))
        print(kp_files)
        # kp_files_total.append(kp_files)

        DESC_PATH = os.path.join(obj_pc_dir,'features')
        desc_files = sorted(glob.glob(DESC_PATH+"/*.npy"))

        encode_descriptor(desc_files, obj_enc_dir, net)


