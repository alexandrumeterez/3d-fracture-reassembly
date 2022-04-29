import argparse
import os
from os import listdir
from os.path import isfile, join
import glob
from turtle import get_poly
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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import shutil


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
# classes = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase',
#            'board', 'clutter']
# class2label = {cls: i for i, cls in enumerate(classes)}
# seg_classes = class2label
# seg_label_to_cat = {}
# for i, cat in enumerate(seg_classes.keys()):
#     seg_label_to_cat[i] = cat

# pdb.set_trace()
def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y
#### Param yet to grasp
experiment_dir = 'log/part_seg/'+ 'pointnet2_part_seg_msg'
visual_dir = experiment_dir + '/visual/'
visual_dir = Path(visual_dir)
visual_dir.mkdir(exist_ok=True)
K = 10  ## num of neighbors to aggregate the descriptors for a keypoint
NUM_CLASSES = 16
NUM_PART =  50
BATCH_SIZE = 1 ## for dev
######################

# KEYPOINT_PATH = "/home/sombit/objects/"   ## get all the keypoint paths.
# PC_PATH="./data/keypoints/3d_fracture_reassmbly-main/data/"    ## get all the pc path
# USIP_TYPE = "_1vN"
PC_PATH = "/home/sombit/object_test/" 
##
###

# feature_dir = "/home/sombit/keypoints/features/"  ### path to store the descriptors.
# triplet_dir = "./data/keypoints/triplets/"

# pc_files= sorted(glob.glob(PC_PATH+OBJECT+USIP_TYPE+"/*.npy"))
# kp_files= sorted(glob.glob(KEYPOINT_PATH+OBJECT+USIP_TYPE+"/*.npy"))

# pdb.set_trace()


def hook_fn(module, input, output):
  # print(module)
  print("------------Input Grad------------")

  for grad in input:
    try:
      print(grad.shape)
    except AttributeError:
      print ("None found for Gradient")

  print("------------Output Grad------------")
  for grad in output:
    try:
      print(grad.shape)
    except AttributeError:
      print ("None found for Gradient")
  print("\n")
  features.append(output)


def get_classifier():
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(NUM_PART, normal_channel=True)
    # classifier = MODEL.get_model(NUM_CLASSES).cuda()              ## load model to gpu vs cpu
    # print(classifier) 
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth',map_location=torch.device(device))
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.eval()
    return classifier

def check_file(PC, KP):
    kp_path, kp_file = os.path.split(KP)
    pc_path, pc_file = os.path.split(PC)
    pc_file_np = pc_file.split('_')[-1]
    if not kp_file == pc_file_np:
        raise IOError('files are not matching: key_point and point_cloud')

def descriptors(pc_file, kp_file, classifier):
    pc = np.load(pc_file)
    kp = np.load(kp_file)
    # print(kp.shape)
    # sigma = kp[:,3]         # saliency score
    kp = kp[:,:3]           # key_points
    points = pc[:,:3]       # xyz coordinates
    dist, idx = get_nbrs(points, kp)
    # rel_w = get_weights(dist)

    ## prepare data in format for the classifier
    batch_data = pc
    torch_data = torch.Tensor(batch_data)
    torch_data = torch_data.unsqueeze(0)
    torch_data = torch_data.permute(0,2,1)

    global features
    features = []
    with torch.no_grad():
        output = classifier(torch_data, to_categorical(torch.tensor(0), NUM_CLASSES))


    feature_chunk = features[0].squeeze(0).permute(1,0)
    features_kp_10 =  feature_chunk[idx]
    print(features_kp_10.shape)
    # features_agg = np.zeros((kp.shape[0], feature_chunk.shape[1]))
    # for i in range(kp.shape[0]):
    #     f = features_kp_10[i]
    #     w = rel_w[i]
    #     f_kp = (f * w[:, None]).sum(axis=0)
    #     features_agg[i, :] = f_kp
    return features_kp_10[:,0,:]

def store_desc(desc, obj_f_dir, kp_file):
    path , file = os.path.split(kp_file)
    print(obj_f_dir)
    
    p = Path(obj_f_dir)
    p.mkdir(exist_ok=True)
    file_name = os.path.join(obj_f_dir,file[:-4])
    # print(p)
    # print(file_name)
    np.save(file_name, desc)

def iterate_over_files(pc_files, kp_files, classifier,obj_f_dir):
    for pc_file, kp_file in zip(pc_files, kp_files):
        # check_file(pc_file, kp_file)
        desc = descriptors(pc_file, kp_file, classifier)
        store_desc(desc, obj_f_dir, kp_file)



## cube_20_FN -->
## cylinder_20_seed_1_FN
# KEYPOINT_PATH = "./data/keypoints/keypoints_2/keypoints_2/"

# obj_f_dir = os.path.join(feature_dir, OBJECT + USIP_TYPE)
# ensure_dir(obj_f_dir)

classifier = get_classifier()
classifier.conv1.register_forward_hook(hook_fn)
# PC_PATH ='/home/sombit/Downloads/3D_Fracture_Reassembly-20220329T110044Z-001/3D_Fracture_Reassembly/Test-Train Sets/5_shapes_4-1_seeds/test/connected_1vN/fragments_1'
# KEYPOINT_PATH ='/home/sombit/keypoints/5_shapes_4-1_seeds/tsf_NFJEKSZE'
## get descriptors


# object_type =os.listdir(PC_PATH)
# get_pointnet = True
# if(get_pointnet ):
#   for object_t in object_type:
#       object_t_dir = os.path.join(PC_PATH,object_t)
#       objects = os.listdir(object_t_dir)
#       for object_ in objects:
#           obj_pc_dir = os.path.join(object_t_dir,object_)
#           obj_f_dir = os.path.join(obj_pc_dir,"features_cls")
#           if os.path.exists(obj_f_dir):
#             shutil.rmtree(obj_f_dir)
          
#           p = Path(obj_f_dir)
#           p.mkdir(exist_ok=True)
#           pc_files= sorted(glob.glob(obj_pc_dir+"/*.npy"))
#           KEYPOINT_PATH = os.path.join(obj_pc_dir,'keypoints')
#           # KEYPOINT_PATH = os.path.join(KEYPOINT_PATH,os.listdir(KEYPOINT_PATH)[0])
#           kp_files= sorted(glob.glob(os.path.join(KEYPOINT_PATH+"/*.npy")))
#           # print(obj_f_dir)
#           iterate_over_files(pc_files, kp_files, classifier,obj_f_dir)


# DESC_PATH = "/home/sombit/keypoints/features/"  ### path to store the descriptors.

# #     ### get triplet data

# desc_files = sorted(glob.glob(DESC_PATH+"/*.npy"))
    # generate_triplets(kp_files, desc_files, obj_triplet_dir)
# if __name__ == '__main__':
#     for object_t in object_type:
#         print(object_t)
#         object_t_dir = os.path.join(PC_PATH,object_t)
#         objects = os.listdir(object_t_dir)
#         for object_ in objects:
#             obj_pc_dir = os.path.join(object_t_dir,object_)
#             obj_triplet_dir = os.path.join(obj_pc_dir,"triplets_cls")
#             pc_files= sorted(glob.glob(obj_pc_dir+"/*.npy"))
#             KEYPOINT_PATH = os.path.join(obj_pc_dir,'keypoints')
#             kp_files= sorted(glob.glob(os.path.join(KEYPOINT_PATH+"/*.npy")))

#             DESC_PATH = os.path.join(obj_pc_dir,'features_cls')
#             # KEYPOINT_PATH = os.path.join(KEYPOINT_PATH,os.listdir(KEYPOINT_PATH)[0])

#             desc_files = sorted(glob.glob(DESC_PATH+"/*.npy"))

            # iterate_over_files(pc_files, kp_files, classifier,obj_f_dir)

            # generate_triplets(kp_files, desc_files, obj_triplet_dir)
# #             break
#         break

