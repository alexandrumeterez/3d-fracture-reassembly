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

def load_data(triplet_files):

    anchor=np.zeros((1,128))
    positive=np.zeros((1,128))
    negative=np.zeros((1,128))

    for obj_file_list in triplet_files:
        for f in obj_file_list:
            dict = np.load(f)
            anc = dict['anchor']
            pos = dict['positive']
            neg = dict['negative']

            anchor = np.append(anchor, anc, axis=0)
            positive = np.append(positive, pos, axis=0)
            negative = np.append(negative, neg, axis=0)
            print(f)
    anchor = np.delete(anchor, 0, axis = 0)
    positive = np.delete(positive, 0, axis = 0)
    negative = np.delete(negative, 0, axis = 0)
    # pdb.set_trace()

    return anchor, positive, negative

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )
        self.layer_norm = torch.nn.LayerNorm(32)


    def forward(self, x):
        # x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        out = self.layer_norm(logits)
        return out

if __name__ == "__main__" :
    names = ['brick', 'cube_6', 'cube_20',  'cake', 'gargoyle', 'head', 'sculpture', 'venus', 'cat', 'cat_low_res_0']
    for i in range(0,6):
        if i == 2: continue
        names.append(f'cat_seed_{i}')
    for i in range (1,15):
        names.append(f'cube_20_seed_{i}')
    for i in range (1,10):
        names.append(f'cylinder_20_seed_{i}')

    # KEYPOINT_PATH = "./data/keypoints/keypoints_2/keypoints_2/"
    KEYPOINT_PATH = "./data/keypoints/keypoints_4/"

    DESC_PATH = "./data/keypoints/features/"
    TRIPLET_PATH = "./data/keypoints/triplets/"

    triplet_files = []
    for name in names[:2]:
        OBJECT = name
        # OBJECT = "cube_6"
        USIP_TYPE = "_FN"
        print("Processing: ", OBJECT)
        kp_files= sorted(glob.glob(KEYPOINT_PATH+OBJECT+USIP_TYPE+"/*.npy"))
        desc_files = sorted(glob.glob(DESC_PATH+OBJECT+USIP_TYPE+"/*.npy"))
        triplet_files.append(sorted(glob.glob(TRIPLET_PATH+OBJECT+USIP_TYPE+"/*.npz")))
    for name in names[:1]:
        OBJECT = name
        # OBJECT = "cube_6"
        USIP_TYPE = "_1vN"
        print("Processing: ", OBJECT)
        kp_files= sorted(glob.glob(KEYPOINT_PATH+OBJECT+USIP_TYPE+"/*.npy"))
        desc_files = sorted(glob.glob(DESC_PATH+OBJECT+USIP_TYPE+"/*.npy"))
        triplet_files.append(sorted(glob.glob(TRIPLET_PATH+OBJECT+USIP_TYPE+"/*.npz")))

    print(triplet_files)
    anchor, positive, negative = load_data(triplet_files)

    lr = 1e-5
    num_epoches = 2000
    net=NeuralNetwork()
    if torch.cuda.is_available() :
        net = net.cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr)
    triplet_loss = nn.TripletMarginLoss(margin=0.1)
    test_only = 0
    l_his=[]
    if test_only==0:

        for epoch in range(num_epoches):
            print('Epoch:', epoch + 1, 'Training...')
            running_loss = 0.0
            # for i,data in range(anchor.shape[0]):
            if torch.cuda.is_available():
                anchor = anchor.cuda()
                positive = positive.cuda()
                negative = negative.cuda()

            optimizer.zero_grad()
            enc_a = net(torch.Tensor(anchor))
            enc_p = net(torch.Tensor(positive))
            enc_n = net(torch.Tensor(negative))

            loss = triplet_loss(enc_a, enc_p, enc_n)
            loss.backward()
            optimizer.step()
            # if i % 20 == 19:
            # pdb.set_trace()
            l_his.append(loss.item())
            # print statistics
            # running_loss += loss.data[0]
            # if i % 100 == 99:
            #     print('[%d, %5d] loss: %.3f' %
            #           (epoch + 1, i + 1, running_loss / 100))
            #     running_loss = 0.0

        print('Finished Training')
        torch.save(net.state_dict(), 'triplet_model')
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.plot(l_his)
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        fig.savefig('plotad.png')
