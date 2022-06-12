# 3D Feature Point Learning for Fractured Object Reassembly

This project consists of multiple pipeline stages. Each toplevel folder inside `/src` contains another ReadMe with the detailed instructions.

## Overview

- `/src/dataset_generation` contains files to generate fractures saved in `.npy` format out of `.obj` 3D Meshes.
- `/src/model` contains the keypoint and descriptor generation in two ways:
    1. Classical Method from https://ieeexplore.ieee.org/document/9279208
    2. Learned Method based on PointNet++
- `/src/optimization` contains the final reassembly optimizer for the fragments with its descriptors. 

## Step by step reassembly tutorial

We provide a step by step tutorial for the Cube object and reassembly.

To extract keypoints and features, run
```
python3 classical_method.py --dataset_dir Cube_dense_8/Cube_8_seed_0/ --n_keypoints 512 --keypoint_radius 0.04 --r_vals 0.04 0.05 0.06 0.08 0.10 --threshold 0.2 --mode 1 --nms --nms_rad 0.04
```

Afterwards, to perform the reconstruction:
```
```
