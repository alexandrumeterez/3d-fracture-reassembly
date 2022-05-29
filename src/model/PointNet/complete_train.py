from cgi import test
import os
from pyexpat import features
from re import L
from black import out 
import numpy as np
import glob
from get_neighbors import *
from model import *
from cv2 import log
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
# from online_triplet_loss.losses import *
from torch.optim.lr_scheduler import ExponentialLR 
from triplet_loss import *
# from find_distances import generate_triplets
import torch.nn as nn

import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
NUM_CLASSES = 16
NUM_PART =  50
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

experiment_dir = 'log/part_seg/'+ 'pointnet2_part_seg_msg'
# model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
model_name = 'pointnet2_part_seg_msg'

# print(model_name)
MODEL = importlib.import_module(model_name)

# def hook_fn(module, input, output):
#       # print(module)
#   print("------------Input Grad------------")

#   for grad in input:
#     try:
#       print(grad.shape)
#     except AttributeError:
#       print ("None found for Gradient")

#   print("------------Output Grad------------")
#   for grad in output:
#     try:
#       print(grad.shape)
#     except AttributeError:
#       print ("None found for Gradient")
#   print("\n")
#   features.append(output)

classifier = MODEL.get_model(NUM_PART, normal_channel=True)
checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth',map_location=torch.device(device))
classifier.load_state_dict(checkpoint['model_state_dict'])
net = NeuralNetwork()
# if torch.cuda.is_available() :
#     net = net.to(device)
classifier.conv2 =net 
# print(classifier)
# print(classifier.conv2)
classifier.to(device)

# for name, param in classifier.named_parameters():
#     # if param.get_device()==device:
#     print (name,param.get_device())
# classifier.conv1.register_forward_hook(hook_fn)
# class fragment ():
#     def __init__(self, pc_file_path) :
#         self.pc_file = pc_file_path
#     def 
#     def kp_index(self,points,kp):
#         dist, idx = get_nbrs(points, kp)
#         self.index = idx

class object():
    def __init__(self,obj_path):
        self.pc_dir = obj_path
        self.kp_dir = os.path.join(self.pc_dir,"keypoints")
        self.data = {}
        self.load()
    def load (self):
        frag_list = sorted(glob.glob(self.pc_dir +"/*.npy")) 
        kp_list   = sorted(glob.glob(self.kp_dir +"/*.npy"))
        
        for i in range(len(frag_list)):
            # print(i)
            self.data[i]={}
            frag = frag_list[i]
            kp = kp_list[i]
            frag_pc = np.load(frag)
            frag_kp = np.load(kp)

            
            dist, idx = get_nbrs(frag_pc[:,:6], frag_kp[:,:3])
            self.data[i]['pc']  = frag_pc[:,:6]
            self.data[i]['kp'] = frag_kp[:,:3]
            self.data[i]['idx'] = idx
            i = i+1

class dataset_pc():
    
    def __init__(self, object_dir):
        self.dataset_dir = object_dir 
        object_type =os.listdir(self.dataset_dir)
        self.obj_path_list = []
        for object_t in object_type:
            obj_path = os.path.join(self.dataset_dir ,object_t)
            self.obj_path_list.append(obj_path)
            # objects = os.listdir(object_t_dir)
            # for object_ in objects:
            #     obj_pc_dir = os.path.join(object_t_dir,object_)

            # triplet_path = os.path.join(self.object_pc_dir,'triplets')
            # files = sorted(glob.glob(triplet_path+"/*.npz"))
            # self.triplets_list.extend(files)
    def __len__(self):
            # print(len(self.triplets_list))

        return len(self.obj_path_list)
    
    def __getitem__(self, idx):

        obj = object(self.obj_path_list[idx])
        return obj.data

#### Params #### 
dataset_path  = "/home/somdey/object_inv/train"
epochs = 10
batch_size = 1





# pc_data = dataset_pc(PC_PATH_train)
# train_dataloader = DataLoader(pc_data, batch_size=batch_size,
#                     shuffle=False, num_workers= 5)
# pc_data_val = dataset_pc(PC_PATH_val)
# val_dataloader = DataLoader(pc_data_val, batch_size=batch_size,
#                     shuffle=False, num_workers= 5)    
dataset_3d  = dataset_pc(dataset_path)
train_size = int(0.8 * len(dataset_3d))
print(len(dataset_3d))
val_size = len(dataset_3d) - train_size
train_set, val_set = torch.utils.data.random_split(dataset_3d, [train_size, val_size])

dataloader_train=torch.utils.data.DataLoader(train_set,batch_size=batch_size, shuffle=True)
dataloader_val = torch.utils.data.DataLoader(val_set,batch_size=1, shuffle=True)

##### Network Params ###
optimizer = torch.optim.AdamW(net.parameters(), lr = 0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
scheduler = ExponentialLR(optimizer, gamma=0.99)
triplet_loss = nn.TripletMarginLoss(margin=0.2)
########## 
def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y
features = []

def positive_abs(dist,positive_rad):
    min_dist = np.min(dist, axis=1)
    idx = np.where(min_dist<positive_rad)
    min_idx = np.argmin(dist, axis=1)
    # print(min_idx[idx])
    return min_dist[idx], min_idx[idx], idx
def negative_abs(dist,f_dist, min_dist, min_idx, idx,negative_rad):
    f_dist_filtered = f_dist
    dist_filtered = dist
    dist_sub = dist_filtered - negative_rad
    dist_sub_1 = np.where(dist_sub>0, f_dist_filtered, np.inf)

    neg_dist = np.min(dist_sub_1, axis=1)
    neg_idx = np.argmin(dist_sub_1, axis=1)
    return neg_dist, neg_idx

def generate_triplets(kp_files, desc_files, obj_triplet_dir,positive_rad,negative_rad):
    for i in range(len(kp_files)):

        ## file name for fragment processed
        frag = kp_files[i]
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
        f_dist = cdist(frag_desc[:,:], other_descs[:,:])
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
        # store_triplet(anchor, positive_desc, negative_desc, obj_triplet_dir, frag)
    # pdb.set_trace()


print("Start Training")
def get_triplet(features, kps,frag_id,rad_pos,rad_neg):
    loss = []
    for frag in kps.keys():
        if (frag != frag_id) :
            dist = torch.cdist(kps[frag_id],kps[frag] )
            min_dist, min_indices = torch.min(dist,axis = 1)
            idx = torch.where(min_dist <rad_pos)
            idx = idx[0]
            
            min_idx = torch.argmin(dist, axis=1)
            pos = min_idx[idx]
            anchor = features[frag_id][idx]
            positive = features[frag][pos]

            f_dist = torch.cdist(features[frag_id],features[frag] )
            #print(dist.type(),f_dist.type(),rad_neg)
            neg_mat = torch.where(dist>rad_neg,f_dist,f_dist*1000000)
            
            min_idx = torch.argmin(neg_mat ,axis = 1)
            neg = min_idx[idx]
            negative = features[frag][neg] 
            if(idx.numel()):
                #print(anchor,positive, negative)
                loss_curr =  triplet_loss(anchor, positive, negative)
                #if(torch.isnan(loss_curr)):
                #    print(anchor,positive,negative,idx)
                loss.append(loss_curr)
    loss_total = sum(loss)/len(loss) 


    return loss_total




    # other_kp = torch.zeros(1,3)
    # for frag in kps.keys():
    #     if  frag != frag_id:
    #         other_kp.append(kp[frag])
    # other_kp = torch.delete(other_kp,0,axis = 0)
    # dist = torch.cdist(kps[frag_id][:,:3], other_kp[:,:3])
    # min_dist = torch.min(dist, axis=1)
    # idx = np.where(min_dist<rad)
    # min_idx = np.argmin(dist, axis=1)
    # # print(min_idx[idx])
    # return min_dist[idx], min_idx[idx], idx

net.float()

for epoch in range(epochs):
        running_loss = 0.0
        optimizer.zero_grad()
        for i_batch, batch in enumerate(dataloader_train):
            features = {}
            kps = {}
            for frag_id in batch.keys():
                
                pc = batch[frag_id]['pc'].float()
                idx = batch[frag_id]['idx']

                torch_data = torch.Tensor(pc).to(device)

                # torch_data = torch_data.unsqueeze(0)
                torch_data = torch_data.permute(0,2,1)

                kp = batch[frag_id]['kp'].float()
                torch_kp = torch.Tensor(kp).to(device)
                kps[frag_id] = torch_kp[0]
               
                output = classifier(torch_data, to_categorical(torch.tensor(0).to(device),NUM_CLASSES))
                # print(output[0][0,idx[0,:,0]].shape)
                features[frag_id] = output[0][0,idx[0,:,0]]
                # print(features[0].shape)
            loss = []
            for frag_id in batch.keys():
                loss.append(get_triplet(features, kps,frag_id, 0.01,0.04))
            loss_sum = sum(loss)/len(loss)
            loss_sum.backward()
            optimizer.step()
            print(loss_sum.item(), " Loss Currently")
        torch.save(classifier.state_dict(), "/home/somdey/point.pth")

