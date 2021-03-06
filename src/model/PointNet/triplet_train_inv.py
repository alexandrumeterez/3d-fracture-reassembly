import argparse
# from msilib import sequence
import os
from os import listdir
from os.path import isfile, join
import glob
from this import d
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
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from online_triplet_loss.losses import *
from torch.optim.lr_scheduler import ExponentialLR 
from triplet_loss import *
from find_distances import generate_triplets
import torch.nn as nn

import torch.nn.functional as F
PC_PATH_train = "/home/sombit/object/train"
PC_PATH_val = "/home/sombit/object/val"



class dataset_pc(Dataset):

    def __init__(self, object_path, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.triplets_list = []
        self.kp_files = []
        self.anc = np.zeros((1,128))
        self.pos = np.zeros((1,128))
        self.neg = np.zeros((1,128))
        object_type =os.listdir(object_path)
        for object_t in object_type:
            # print(object_t)
            self.object_pc_dir = os.path.join(object_path,object_t)
            # objects = os.listdir(object_t_dir)
            # for object_ in objects:
            #     obj_pc_dir = os.path.join(object_t_dir,object_)

            triplet_path = os.path.join(self.object_pc_dir,'triplets')
            files = sorted(glob.glob(triplet_path+"/*.npz"))
            self.triplets_list.extend(files)
        for obj_file in self.triplets_list:
            dict = np.load(obj_file)
            anc = dict['anchor']
            pos = dict['positive']
            neg = dict['negative']
            self.anc = np.append(self.anc, anc, axis=0)
            self.pos = np.append(self.pos, pos, axis=0)
            self.neg = np.append(self.neg, neg, axis=0)
        self.anc = np.delete(self.anc, 0, axis = 0)
        self.pos = np.delete(self.pos, 0, axis = 0)
        self.neg = np.delete(self.neg, 0, axis = 0)
       


        # anchor, positive, negative = load_data(triplet_files)

    def __len__(self):
        # print(len(self.triplets_list))

        return self.anc.shape[0]

    def load_data(self,triplet_files):
        
        anchor=np.zeros((1,128))
        positive=np.zeros((1,128))
        negative=np.zeros((1,128))

        for obj_file_list in triplet_files:
            # for f in obj_file_list:
            dict = np.load(obj_file_list)
            anc = dict['anchor']
            pos = dict['positive']
            neg = dict['negative']

            anchor = np.append(anchor, anc, axis=0)
            positive = np.append(positive, pos, axis=0)
            negative = np.append(negative, neg, axis=0)
                # print(f)
        anchor = np.delete(anchor, 0, axis = 0)
        positive = np.delete(positive, 0, axis = 0)
        negative = np.delete(negative, 0, axis = 0)
        # pdb.set_trace()

        return anchor, positive, negative

    def __getitem__(self, idx):

        data = {'anchor': self.anc[idx], 'positive': self.pos[idx], 'negative': self.neg[idx]}
        return data
        

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
            # print(f)
    anchor = np.delete(anchor, 0, axis = 0)
    positive = np.delete(positive, 0, axis = 0)
    negative = np.delete(negative, 0, axis = 0)
    # pdb.set_trace()

    return anchor, positive, negative    
def load_kp(kp_files):
    kp = np.zeros((1,3))
    for f in kp_files:
        for f_ in f:
            dict = np.load(f_)
            kp_ = dict[:,:3]
            kp = np.append(kp, kp_, axis=0)
    kp = np.delete(kp, 0, axis = 0)
    return kp

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128)

        )
        # self.layer_norm = F.normalize(x, dim = 0)


    def forward(self, x):
        # x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        # out = self.layer_norm(logits)
        out = F.normalize(logits, dim = 0)
        return out

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
        # print("Storing encodings for fragment: ", f)
        # pdb.set_trace()
        store_descriptors(enc_desc.cpu().detach().numpy(), obj_encoded_desc_dir, f)

def save_encoded(dataset_path,feature_folder,encoded_folder):
    net_1 = NeuralNetwork().to('cuda')
    net_1.load_state_dict(torch.load('triplet_model.pth'))
    net_1.eval()
    object_type =os.listdir(PC_PATH_train)
    for object_ in object_type:
        obj_pc_dir = os.path.join(dataset_path,object_)
        obj_enc_dir = os.path.join(obj_pc_dir,encoded_folder)
        

        DESC_PATH = os.path.join(obj_pc_dir,features_folder)
        # KEYPOINT_PATH = os.path.join(KEYPOINT_PATH,os.listdir(KEYPOINT_PATH)[0])

        desc_files = sorted(glob.glob(DESC_PATH+"/*.npy"))
        # print(obj_enc_dir)
        encode_descriptor(desc_files, obj_enc_dir, net_1)

keypoints_folder = 'keypoints'
features_folder_inv = 'feature_contact_inv'
features_folder = 'features'
triplets_folder = 'triplets'
encoded_folder = 'encoded'
encoded_folder_inv = 'encoded_inv'

def generate_new_trips(dataset_path):
    object_type =os.listdir(dataset_path)
        # print(object_t)
    for object in object_type:
        obj_pc_dir = os.path.join(PC_PATH_train,object)
        obj_feature_dir = os.path.join(obj_pc_dir,features_folder)
        obj_triplet_dir = os.path.join(obj_pc_dir,triplets_folder)

        pc_files= sorted(glob.glob(obj_pc_dir+"/*.npy"))
        keypoint_path = os.path.join(obj_pc_dir,keypoints_folder)
        kp_files= sorted(glob.glob(os.path.join(keypoint_path+"/*.npy")))
        desc_path = os.path.join(obj_pc_dir,encoded_folder)

        desc_files = sorted(glob.glob(desc_path+"/*.npy"))
        desc_path_inv = os.path.join(obj_pc_dir,encoded_folder_inv)
        desc_files_inv = sorted(glob.glob(desc_path_inv+"/*.npy"))
        # print(desc_path)
        generate_triplets(kp_files, desc_files,desc_files_inv, obj_triplet_dir,positive_rad=0.001,negative_rad=0.04)

def get_achor_pos():

    triplet_files = []
    kp_files = []
    object_type =os.listdir(PC_PATH_train)
    for object_t in object_type:
        # print(object_t)
        object_t_dir = os.path.join(PC_PATH_train,object_t)
        objects = os.listdir(object_t_dir)
        for object_ in objects:
            obj_pc_dir = os.path.join(object_t_dir,object_)

            triplet_path = os.path.join(obj_pc_dir,'triplets')
            triplet_files.append(sorted(glob.glob(triplet_path+"/*.npz")))
    anchor, positive, negative = load_data(triplet_files)
    kps = load_kp(kp_files)
    return anchor,positive,negative,kps

if __name__ == "__main__" :


    num_epoches = 40
    net=NeuralNetwork()
    # if torch.cuda.is_available() :
    #     net = net.cuda()
    net.cuda()
   
    optimizer = torch.optim.AdamW(net.parameters(), lr = 0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
    scheduler = ExponentialLR(optimizer, gamma=0.99)
    triplet_loss = nn.TripletMarginLoss(margin=0.15)
    train_now = True
    test_now = True
    batch_size  = 100
    l_his=[]
    best = 100.0
    net = net.float()


  
    print("loading data")
    pc_data = dataset_pc(PC_PATH_train)
    train_dataloader = DataLoader(pc_data, batch_size=batch_size,
                        shuffle=True, num_workers= 2)
    pc_data_val = dataset_pc(PC_PATH_val)
    val_dataloader = DataLoader(pc_data_val, batch_size=100,
                        shuffle=True, num_workers= 2)    
    # pc_data_test = dataset_pc(PC_PATH_test)
    # test_dataloader = DataLoader(pc_data_test, batch_size=100,
    #                     shuffle=True, num_workers= 2)                                        
    print("data loaded" , len(train_dataloader), len(val_dataloader))
    print("Starting training")
    loss_hist = {}
    loss_hist["train"] = []
    loss_hist["val"] = []
    loss_hist["test"] = []
    
    if train_now :
        net.train()
        for epoch in range(num_epoches):
            running_loss = 0.0
            for i_batch, batch in enumerate(train_dataloader):

                
                optimizer.zero_grad()

                enc_a = net(batch['anchor'].float().cuda())
                enc_p = net(batch['positive'].float().cuda())
                enc_n = net(batch['negative'].float().cuda())


                loss = triplet_loss(enc_a, enc_p, enc_n)
                # print(loss.item())
            # loss, pos_mask, neg_mask = online_mine_hard(kps[:,0],enc_a,margin = 0.1,device ='cuda')

                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.5)
                optimizer.step()
                scheduler.step()
                running_loss += loss.item()

            print("Epoch", epoch ," Training_Loss", running_loss/len(train_dataloader))
            loss_hist["train"].append(running_loss/len(train_dataloader))
            print("Validation")
            running_loss_val = 0.0
            for i_batch, batch in enumerate(val_dataloader):
                net.eval()
                
                # for i,data in range(anchor.shape[0]):
                # if torch.cuda.is_available():
                
                optimizer.zero_grad()
                enc_a = net(batch['anchor'].float().cuda())
                enc_p = net(batch['positive'].float().cuda())
                enc_n = net(batch['negative'].float().cuda())


                loss = triplet_loss(enc_a, enc_p, enc_n)
                running_loss_val += loss.item()
            print(running_loss_val/len(val_dataloader))
            loss_hist["val"].append(running_loss_val/len(val_dataloader))

            torch.save(net.state_dict(), 'triplet_model.pth')
            if(running_loss_val/len(val_dataloader) <=best):
                torch.save(net.state_dict(), 'triplet_model_best.pth')
                best = running_loss/len(train_dataloader)
            save_encoded(PC_PATH_train, features_folder,encoded_folder)
            save_encoded(PC_PATH_val, features_folder,encoded_folder)

            generate_new_trips(PC_PATH_train)
            generate_new_trips(PC_PATH_val)
            
            pc_data = dataset_pc(PC_PATH_train)
            train_dataloader = DataLoader(pc_data, batch_size=batch_size,shuffle=True, num_workers=2)
            pc_data = dataset_pc(PC_PATH_val)
            val_dataloader = DataLoader(pc_data, batch_size=batch_size,shuffle=True, num_workers=2)



        print('Finished Training')
        torch.save(net.state_dict(), 'triplet_model.pth')
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.plot(loss_hist["train"], label="train")
        ax.plot(loss_hist["val"], label="val")
        
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        fig.savefig('loss_plot.png')
