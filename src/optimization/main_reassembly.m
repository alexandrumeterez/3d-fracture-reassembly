% Description: Main function for 3D reassembly
% --------------------------------------------
% Semester project 3D vision lecture
% 3D Feature Point Learning for Fractured Object Reassembly
% Team: Du Chaoyu (chaoyu.du@arch.ethz.ch)
% Ulrich Steger (ulsteger@student.ethz.ch)
% Eshaan Mudga (emudgal@student.ethz.ch) 
% Florian Hürlimann (fhuerliman@student.ethz.ch)
% Author of this code: F. Huerlimann

%% Clean up
clear;
clc;

%% Step 1: Import data
% Manually configure input data directories in "import_3d.m" (config
% section)
% Import of fragment point clouds, keypoints, feature descriptors
[f,kp,ft] = import_3d;

%% Step 2: Find keypoint pairs with ground truth
% Note: This step is only necessary if keypoint pair detection based on
% feature descriptors is not working. 
% Input parameters: Define knn radius (depends on model dimensions)
% Important (!): For cube_6 use 0.08, for brick use 0.5.
%[f,kp,kp_index,kp_corr,frag_corr] = comp_kp(f,kp,knn_r)
[f,kp,kp_index,kp_corr,frag_corr] = comp_kp(f,kp,0.08);

%% Step 3: Perform reassembly
% Note: Check config section of "assy_3d.m"
[f_out, kp_out] = assy_3d(f,kp,kp_corr,frag_corr);

