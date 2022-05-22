# get all the objects files in this list and then iterate over them

from tkinter.tix import Tree
import numpy as np 
import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from get_neighbors import *
from find_distances import *
import importlib
from tqdm import tqdm
import provider
import numpy as np
import pdb
import sys 
import shutil


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

experiment_dir = 'log/part_seg/'+ 'pointnet2_part_seg_msg'
visual_dir = experiment_dir + '/visual/'
visual_dir = Path(visual_dir)
visual_dir.mkdir(exist_ok=True)
# K = 10  ## num of neighbors to aggregate the descriptors for a keypoint
NUM_CLASSES = 16
NUM_PART =  50


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
def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y

def descriptors(pc_file, kp_file, classifier):
    pc = np.load(pc_file)
    kp = np.load(kp_file)
    # print(kp.shape)
    # sigma = kp[:,3]         # saliency score
    kp = kp[:,:3]           # key_points
    points = pc[:,:6]       # xyz coordinates
    # points = points[:,3:]*-1
    dist, idx = get_nbrs(points, kp)
    # rel_w = get_weights(dist)

    ## prepare data in format for the classifier
    batch_data = pc
    torch_data = torch.Tensor(batch_data)
    torch_data = torch_data.unsqueeze(0)
    torch_data = torch_data.permute(0,2,1)
    torch_data[:,:3,:]=torch.nn.functional.normalize(torch_data[:,:3,:], p=1.0, dim = 2)
    print(torch_data)

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

object_paths_list = ['/home/sombit/object_inv/train/','/home/sombit/object_inv/val/'] 
keypoints_folder = 'keypoints'
features_folder= 'features'

# features_folder = 'features'
triplets_folder = 'triplets'
encoded_folder = 'encoded'
store_feature = True
store_triplet = True
# model_path = 'triplet_model_new.pth'

classifier = get_classifier()
classifier.conv1.register_forward_hook(hook_fn)

## store pointnet descriptor for each object 
for object_path in object_paths_list:
    object_type =os.listdir(object_path) ########## get all the objects in this list
    print(object_path, object_type)

    if(store_feature):
        for object in object_type:
            obj_pc_dir = os.path.join(object_path,object)
            obj_feature_dir = os.path.join(obj_pc_dir,features_folder)
            if os.path.exists(obj_feature_dir):
                shutil.rmtree(obj_feature_dir)
            p = Path(obj_feature_dir)
            p.mkdir(exist_ok=True)
            
            pc_files= sorted(glob.glob(obj_pc_dir+"/*.npy"))
            keypoint_path = os.path.join(obj_pc_dir,keypoints_folder)
            # KEYPOINT_PATH = os.path.join(KEYPOINT_PATH,os.listdir(KEYPOINT_PATH)[0])
            kp_files= sorted(glob.glob(os.path.join(keypoint_path+"/*.npy")))
            # print(obj_f_dir)
            iterate_over_files(pc_files, kp_files, classifier,obj_feature_dir)
    if(store_triplet):
        for object in object_type:
            obj_pc_dir = os.path.join(object_path,object)
            obj_feature_dir = os.path.join(obj_pc_dir,features_folder)
            obj_triplet_dir = os.path.join(obj_pc_dir,triplets_folder)

            pc_files= sorted(glob.glob(obj_pc_dir+"/*.npy"))
            keypoint_path = os.path.join(obj_pc_dir,keypoints_folder)
            kp_files= sorted(glob.glob(os.path.join(keypoint_path+"/*.npy")))
            desc_path = os.path.join(obj_pc_dir,features_folder)

            desc_files = sorted(glob.glob(desc_path+"/*.npy"))

            # desc_path_inv = os.path.join(obj_pc_dir,features_folder_inv)
            # desc_files_inv = sorted(glob.glob(desc_path_inv+"/*.npy"))
            # print(desc_files,desc_files_inv)
            generate_triplets(kp_files, desc_files, obj_triplet_dir,positive_rad=0.001,negative_rad=0.04)


