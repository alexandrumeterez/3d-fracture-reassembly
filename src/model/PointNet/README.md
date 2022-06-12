
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

- Data 
    - obj1 
      - keypoints 
      - *.npy files (point cloud data)
    - obj2 
      - keypoints 
      - *.npy files (point cloud data)
    - ...
***
## Steps to train: 
1) Run the generate_features.py file, creates pointnet features folder, calculates triplets using these features in the triplets folder 
```
python generate_features.py
```

2) Run the triplets_train.py file to train.
```
python triplets_train.py
```


## Steps for complete train:  
Requires downsampled point cloud data.
```
python complete_train.py
```

