# 3D Feature Point Learning for Fractured Object Reassembly

This project consists of multiple pipeline stages. Each toplevel folder inside `/src` contains another ReadMe with the detailed instructions.

## Overview

- `/src/dataset_generation` contains files to generate fractures saved in `.npy` format out of `.obj` 3D Meshes.
- `/src/model` contains the keypoint and descriptor generation in two ways:
    1. Classical Method from https://ieeexplore.ieee.org/document/9279208
    2. Learned Method based on PointNet++
- `/src/optimization` contains the final reassembly optimizer for the fragments with its descriptors. 

## Step by step reassembly tutorial
