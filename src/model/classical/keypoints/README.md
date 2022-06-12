# Classical features and keypoints

In this folder we have the code for keypoint detection and feature extraction based on https://ieeexplore.ieee.org/document/9279208. In addition, we have visualization code that we used to generate the figures in the paper.

## How to run?

The tasks that can be done with the code in this directory are keypoint extraction (using SD distance), classical feature extraction, visualization, subsampling.

### Keypoint extraction

Using the `extract_keypoints.py` script. 

To extract keypoints for a whole directory of `.npy` files (a full fragment):
```
python extract_keypoints.py --object_dir OBJECT_DIR --nkeypoints NKEYPOINTS --r R --save
```

To extract keypoints for a single shard (one `.npy` file):
```
python extract_keypoints.py --object_dir SHARD_FILE --nkeypoints NKEYPOINTS --r R --save --file
```

### Feature extraction (using the clasical method from the paper)

Using the `classical_method.py` script. Parameter explanation:

```
--num_features: choose how many of the 7 features to extract
--keypoint_radius: radius to choose for the SD distance keypoint extraction
--n_keypoints: how many keypoints to keep
--r_vals: the multiple radius values used by the feature extractor
--threshold: (only used for visualization) threshold to consider 2 feature matrices close enough for a match

Edge filtering (https://arxiv.org/pdf/1809.10468.pdf)
--k: used for filtering edge points 
--lbd: tolerance for edge filtering

Different edge filtering:
--nms: Whether to use nms for filtering edge points
--nms_rad: NMS radius
```

To extract features for a whole directory of `.npy` files - where you can pass all the params from above:

```
python classical_method.py --mode 1 --dataset_dir DATASET_DIR
```

### Visualize matches between 2 fragments
Using the `classical_method.py` script.
```
python classical_method.py --mode 2 --fragment1 PATH_TO_F1 --fragment2 PATH_TO_F2
``` 

### Subsampling (using voxel subsampling)
We can do voxel subsampling on the objects to reduce the dimensionality using `subsample.py`.

```
python subsample.py --dataset-dir DATASET_DIR --voxel-size VOXEL_SIZE
``` 
