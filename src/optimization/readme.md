## Reassembly Codebase (MATLAB)

### 1. Software Requirements
- MATLAB R2021a 
- MATLAB Computer Vision Toolbox ([https://www.mathworks.com/products/computer-vision.html](url))

### 2. Workflow
- `main_reassembly.m` contains 4 blocks which need to be executed sequentially to perform the reassembly. Alternatively run this script to perform the reassembly. We describe the individual blocks and the relevant parameters to be configured below-
  1. The first block (Step 0)is to clear and clean the workspace. Additonally the parameter `L_desc_flag` needs to be set to `1` if you are using the learned descriptors for the reassembly.
  2. The second block (Step 1) is for importing the data, namely the fragments, keypoints and descriptors. For the classical feature-based method, the keypoints and descriptors are located in the same npy file while for the learning-based method, the keypoints and descriptors are present in seperate files. The directories for the fragments and keypoints need to be set in the `import_3d_classical.m` file if using the classical features. Alternatively if using the learned features, the directories of the fragments, keypoints and descriptors need to be set in the `import_3d.m` file. On importing the data, the plots for groundtruth and random pose of fragments along with the keypoints will pop up.
  3. The third block (Step 2) is for establishing the ground-truth correspondences between the keypoints of the fragments. This block needs to be executed only if you want to use groundtruth correspondences for the reassembly, otherwise this block can be skipped. The radius parameter for the GT matching needs to be set for this block. We recommend using the radius of 0.01 for the synthetic datasets and 0.03 for the real-world datasets.
  4. The final block (Step 3) is for carrying out the reassembly. Numerous parameters need to be set which have been described below. On executing this block, the pose graph and the reassembled fragments along with the groundtruth reassembly will appear.
**Reassembly Parameters:**  
Following parameters need to be set in `assy_3d_ransac.m`:
    - `plot_poses_flag` = 0 % Plot all pairwise poses
    - `plot_result_flag` = 1 % Plot graph, assembly
    - `use_ground_truth` = 0 % keypoint pair selection based on ground truth
    - `learned_desc_flag` = 0 % Set to 1 while using Learned descriptors
    - `add_isolated_fragments` = 1 % add fragments without triple match  
Following parameters need to be set in `get_descriptor_pairs_classical.m`:
    - `max_pairs` = 30 % 
    - `max_d` = 0.5 % Threshold value. Pairs with d>max_d will be discarded (Use 0.5 for experiments with Noise. For noise-free datasets use 0.2)
    - `desc_features` = 3 % No of geometric features (3 or 7)
    - `desc_scales` = 5 % No of radii (multi-scale)
  
    
