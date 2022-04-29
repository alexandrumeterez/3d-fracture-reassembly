
# Point net + encoder

1. genearte_features.py 
    * generate and stores pointnet features in features folder   
    * generate and stores triplets using these pointnet features as descriptors

2 triplets_train.py  
* dataloader -> over all keypoints as datapoints.  
* calculate triplet, train for 5 -10 epochs, recalculate the dataloader
