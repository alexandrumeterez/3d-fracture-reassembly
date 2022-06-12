
# Point net + encoder

1. generate_features.py 
    * generate and stores pointnet features in features folder   
    * generate and stores triplets using these pointnet features as descriptors

2 triplets_train.py  
* dataloader -> over all keypoints as datapoints.  
* calculate triplet, train for 5 -10 epochs, recalculate the dataloader

Dataset format required : 

- Data \
    - obj1 \
      - keypoints \
      - *.npy files (point cloud data)
    - obj2 \
      - keypoints 
      - *.npy files (point cloud data)\
    - 
     
Steps to train: 
1) RUN the generate_features.py file, creates pointnet features folder, caculates triplets using these features in the triplets folder 
2)  RUN the triplets_train.py file to train.

Steps for complete train: 
1) >complete_train.py
