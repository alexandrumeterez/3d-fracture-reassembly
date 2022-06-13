
# PointNet++ Encoder
## Code Structure Overview

1. `generate_features.py`
    * generates and stores PointNet++ features in `features` folder    
    * generates and stores triplets using the extracted features as descriptors
2. `triplets_train.py` 
    * dataloader: generates the pytorch dataloader with only keypoints and its descriptors for all the objects in the dataset
    * updates the triplet after every epoch, recalculate the dataset of triplets using the new MLP weights
***
## Setup 
Use the environment.yml file in the conda environment
```
conda env create -f environment.yml
```

## Dataset format: 
The dataset format for `triplet_train.py` is given below. <br>
Sample dataset with features with the MLP model weights which is required in the further steps below: https://drive.google.com/drive/folders/1hvLwhv-7yK5eBSL8z3WQLFZ1KLmY6UMj?usp=sharing
```
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
```
***

Split the dataset into train and val dataset.

## Steps to train: 
In this training approach we first extract the pointnet features (implying frozen pointnet weights) and train the MLP layers independantly. 
1) Run the `generate_features.py` file to create the PointNet++ features folder  and calculate (and save) triplets using these features in the triplets folder
```
python generate_features.py
```

2) Run the `triplets_train.py` file to train.
```
python triplets_train.py
```

Parameters:
- r_p: radius for positive keypoint matches (default 0.001)
- r_n: radius for negative keypoints (default 0.04)
- PC_train: absolute dataset path 
- epochs
- margin
- r_recalculate 

## Steps for inference:
During inference, the script `encode_descripters.py` will be required. 
1) Run `generate_features.py` to obtain the PointNet++ features . The script expects the same data format as above under the test folder
2) Run `encode_descriptors.py` with the correct model_weights (`.pth file` , in the drive link above) and PC_PATH to obtain the descriptors in the encoded folder
***

## Steps for complete train:
This approach is used to train both the PointNet++ and MLP model end-to-end.
The complete_train pipeline is written with test_train split. The requried data format is given below.

```
- Data 
   - obj1 
      - keypoints 
      - *.npy files (point cloud data)
    - obj2 
      - keypoints 
      - *.npy files (point cloud data)
    - ... 
```
   
Requires downsampled point cloud data and the corresponding keypoints.
```
python complete_train.py
```

