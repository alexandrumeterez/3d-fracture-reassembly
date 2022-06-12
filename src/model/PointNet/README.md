
# Point net + encoder
## Code Structure Overview

1. generate_features.py 
    * generate and stores pointnet features in features folder    
    * generate and stores triplets using these pointnet features as descriptors

2. triplets_train.py  
    * dataloader -> over all keypoints as datapoints.  
    * Updates the triplet after every epoch, recalculate the dataset of triplets using the new MLP weights.
***
## Setup 
Use the environment.yml file in the conda environment
```
conda env create -f environment.yml
```
## Dataset format required : 
The dataset format for triplet_train.py is given below. 

- Data 
   - train
       - obj1 
         - keypoints 
         - *.npy files (point cloud data)
       - obj2 
         - keypoints 
         - *.npy files (point cloud data)
       - ...
    - val
       - obj1 
         - keypoints 
         - *.npy files (point cloud data)
       - obj2 
         - keypoints 
         - *.npy files (point cloud data)
       - ...
***
Split the dataset into train and val dataset.

## Steps to train: 
1) Run the generate_features.py file, creates pointnet features folder, calculates triplets using these features in the triplets folder 
```
python generate_features.py
```

2) Run the triplets_train.py file to train.
```
python triplets_train.py
```
Parameters:
- r_p radius for positive keypoint matches , default 0.001
- r_n radius for negative keypoints , default 0.04
- PC_train absolute dataset path 
- epochs, margin, r_recalculate 

***

## Steps for complete train:  
The complete_train pipeline is written with test_train split. The requried data format is given below.
- Data 
   - obj1 
      - keypoints 
      - *.npy files (point cloud data)
    - obj2 
      - keypoints 
      - *.npy files (point cloud data)
    - ... 
   
Requires downsampled point cloud data and the corresponding keypoints.
```
python complete_train.py
```

