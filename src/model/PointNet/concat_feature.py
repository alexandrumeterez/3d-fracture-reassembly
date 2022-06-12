import os
from pyexpat import features
from turtle import shape 
import numpy as np
obj_path = '/home/sombit/object_new/'
segs = os.listdir(obj_path)
for seg in segs:
    obj_files = os.path.join(obj_path,seg)
    for object in os.listdir(obj_files):
        obj = os.path.join(obj_files,object)
        features_files = os.path.join(obj,'features_cls')
        keypoints = os.path.join(obj,'keypoints')
        save_path = os.path.join(obj,'feature_contact')
        os.makedirs(save_path,exist_ok=True)
        print(save_path)
        for files in os.listdir(features_files):
            # print(files)
            feature = np.load(os.path.join(features_files,files))
            keypoint = np.load(os.path.join(keypoints,files))
            k  = np.hstack([feature[:,:] ,keypoint[:,6:]] )
            print(np.isnan(np.sum(k)),np.isnan(np.sum(feature)),np.isnan(np.sum(keypoint)))
            np.save(os.path.join(save_path,files),k)
