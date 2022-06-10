% Description: Main function for 3D reassembly
% --------------------------------------------
% Semester project 3D Vision 2022
% 3D Feature Point Learning for Fractured Object Reassembly
% Team: Manthan Patel (patelm@student.ethz.ch)
% Sombit Dey (somdey@student.ethz.ch)
% Alexandru Meterez (ameterez@student.ethz.ch) 
% Adrian Hartmann (haadrian@student.ethz.ch)
% Author of this code: Manthan Patel
% Original Code author: Florian Hürlimann

%% Clean up
clear;
clc;
close all;

%Flag to specify Learned Descriptors or Classical Descriptors
%Set Flag to 1 while using learned descriptors data
L_desc_flag = 0;

%% Step 1: Import data
% Manually configure input data directories in "import_3d.m"/
% "import_3d_classical.m"  (config section)
% Import of fragment point clouds, keypoints, feature descriptors

if L_desc_flag
    [f,kp,ft] = import_3d;
else
    [f,kp] = import_3d_classical;
end


%% Step 2: Find keypoint pairs with ground truth
% Note: This step is only necessary if keypoint pair detection based on
% feature descriptors is not working. 
% Input parameters: Define knn radius (depends on model dimensions)
% Important (!): For Syntehtic Datasets use 0.01 and real world datasets
% use 0.03

[f,kp,kp_index,kp_corr,frag_corr] = comp_kp(f,kp,0.01);

%% Step 3: Perform reassembly
% Note: Check config section of "assy_3d.m"
% Note: Check config section of "get_descriptors_pairs_classical.m" to set 
% the descriptor information

if ~exist('kp_corr','var')
   kp_corr = [];
   frag_corr = [];
end

[f_out, kp_out] = assy_3d_ransac(f,kp,kp_corr,frag_corr);

