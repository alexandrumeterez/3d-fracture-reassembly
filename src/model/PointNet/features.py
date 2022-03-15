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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

experiment_dir = 'log/part_seg/'+ 'pointnet2_part_seg_msg'
visual_dir = experiment_dir + '/visual/'
visual_dir = Path(visual_dir)
visual_dir.mkdir(exist_ok=True)
K = 10  ## num of neighbors to aggregate the descriptors for a keypoint
NUM_CLASSES = 16
NUM_PART =  50
BATCH_SIZE = 1 ## for dev


def hook_fn(module, input, output):
  print(module)
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
    print(classifier)
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

    sigma = kp[:,3]         # saliency score
    kp = kp[:,:3]           # key_points
    points = pc[:,:3]       # xyz coordinates
    dist, idx = get_nbrs(points, kp)
    rel_w = get_weights(dist)

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
    features_agg = np.zeros((kp.shape[0], feature_chunk.shape[1]))
    for i in range(kp.shape[0]):
        f = features_kp_10[i]
        w = rel_w[i]
        f_kp = (f * w[:, None]).sum(axis=0)
        features_agg[i, :] = f_kp


    return features_agg

def store_desc(desc, obj_f_dir, kp_file):
    path , file = os.path.split(kp_file)
    file_name = os.path.join(obj_f_dir,file)
    np.save(file_name, desc)

def iterate_over_files(pc_files, kp_files, classifier):
    for pc_file, kp_file in zip(pc_files, kp_files):
        # print(pc_file, kp_file)
        check_file(pc_file, kp_file)
        desc = descriptors(pc_file, kp_file, classifier)
        store_desc(desc, obj_f_dir, kp_file)



## cube_20_FN -->
## cylinder_20_seed_1_FN
# KEYPOINT_PATH = "./data/keypoints/keypoints_2/keypoints_2/"
KEYPOINT_PATH = "./data/keypoints/keypoints_4/"
# PC_PATH="./data/keypoints/real_data_npy/npy/" ## pointcloud path for real data
PC_PATH="./data/keypoints/3d_fracture_reassmbly-main/data/"
OBJECT="cube_6"
USIP_TYPE = "_FN"

feature_dir = "./data/keypoints/features/"
obj_f_dir = os.path.join(feature_dir, OBJECT + USIP_TYPE)
ensure_dir(obj_f_dir)

pc_files= sorted(glob.glob(PC_PATH+OBJECT+"/*.npy"))
kp_files= sorted(glob.glob(KEYPOINT_PATH+OBJECT+USIP_TYPE+"/*.npy"))

# pdb.set_trace()

pc_file=pc_files[0] ## iterate over all the files
kp_file=kp_files[0]


classifier = get_classifier()
classifier.conv1.register_forward_hook(hook_fn)
iterate_over_files(pc_files, kp_files, classifier)
