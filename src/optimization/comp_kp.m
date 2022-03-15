function [f,kp,kp_index,kp_corr,frag_corr] = comp_kp(f,kp,knn_r)
% Descripton: Find keypoint neighbours on other fragments
% Input: f (fragments), kp (keypoints), knn_r (radius for neighbor search)
% knn_r: cube_6 = 0.08, brick = 0.5 (depends on model dimensions)
% Output: f, kp, kp_index (table of all keypoints), 
% kp_corr (correlation matrix of all keypoints)
% frag_corr (keypoint correlation matrix with fragment IDs)
% -------------------------------
% Semester project 3D vision lecture
% 3D Feature Point Learning for Fractured Object Reassembly
% Team: Du Chaoyu (chaoyu.du@arch.ethz.ch)
% Ulrich Steger (ulsteger@student.ethz.ch)
% Eshaan Mudga (emudgal@student.ethz.ch) 
% Florian Hürlimann (fhuerliman@student.ethz.ch)
% Author of this code: F. Huerlimann

% Config
% ------
verbose_flag = 1;
plot_flag = 1;

% Verbose output function
function vdisp(mystring)
    if verbose_flag == true
        format short g;
        disp(mystring); 
    end
end

if nargin < 3
    % default value
    knn_r = 0.05; % Radius for knn search of adjacent fragment points
end
vdisp(['knn radius: ', num2str(knn_r)]);


% Init 3D keypoints
% -----------------
for k=1:length(kp)    
    % Init number of neighbors
    kp{k}.gt.neighbors = zeros(size(kp{k}.gt.x));
end

% Create keypoints index table
% Array with columns: kp_ID, fragment_ID, x, y, z, sigma
kp_index = [];
kp_id = 0;
for k=1:length(kp)
    for i=1:length(kp{k}.gt.x)
        % Column index
        kp_id = kp_id + 1;

        % Insert data into keypoint index table
        kp_index(kp_id,1) = kp_id;
        kp_index(kp_id,2) = k;
        kp_index(kp_id,3) = kp{k}.gt.x(i);
        kp_index(kp_id,4) = kp{k}.gt.y(i);
        kp_index(kp_id,5) = kp{k}.gt.z(i);
        kp_index(kp_id,6) = kp{k}.gt.sigma(i);
        
        % Add keypoint ID to keypoint
        kp{k}.gt.kp_id(i) = kp_id;
        kp{k}.rp.kp_id(i) = kp_id;
    end
end

% Create keypoint correspondency matrix (int = number of correspondencies)
kp_corr = zeros(length(kp_index));
frag_corr = zeros(length(kp_index));

% Loop through fragments
for k=1:length(kp)
    vdisp('==============');
    vdisp(['Find keypoints adjacent to fragment ', num2str(k), ' (', kp{k}.filename, ')']);
    vdisp(['Total keypoints in fragment: ', num2str(length(kp{k}.gt.x))]);
    
    % Loop through keypoints
    for i=1:length(kp{k}.gt.x)
        keypoint = [kp{k}.gt.x(i), kp{k}.gt.y(i), kp{k}.gt.z(i)];
   
        % Loop through other fragements (exclude fragment k)
        for j=1:length(kp)
           % Exclude fragment k
           if j~=k
               
               ptCloud = pointCloud([kp{j}.gt.x, kp{j}.gt.y, kp{j}.gt.z]);
               [indices] = findNeighborsInRadius(ptCloud,keypoint,knn_r);
               if ~isempty(indices)
                   % There are 1 or more neighbor keypoints ...
                   vdisp('--------------');
                   vdisp(['Keypoint ', num2str(i), ': ',num2str(length(indices)), ' neighbors in fragment ', num2str(j)]);
                   vdisp('keypoint ID | fragment ID | x | y | z | sigma');
                   vdisp(kp_index(kp{k}.gt.kp_id(i),:))
                   vdisp('Neighbor keypoints:');
                   vdisp(kp_index(kp{j}.gt.kp_id(indices),:))
                   
                   % Get delta sigmas of neighbor keypoints
                   delta_sigma_list = abs(kp{j}.gt.sigma(indices) - kp{k}.gt.sigma(i));
                   
                   [min_pos] = find(delta_sigma_list == min(delta_sigma_list));
                   % If multiple minima, chose first
                   if length(min_pos) > 1
                       min_pos = min_pos(1);
                   end
                   
                   vdisp(['Optimal neighbor (min delta sigma): ', num2str(min_pos), ...
                       ', delta sigma: ', num2str(delta_sigma_list(min_pos))]);
                   
                   % Add keypoints to correspondency matrix
                   row_index = kp{k}.gt.kp_id(i);
                   col_index = kp{j}.gt.kp_id(indices);
                   kp_corr(row_index, col_index) = kp_corr(row_index, col_index) + 1;
                   kp{k}.gt.neighbors(i) = kp{k}.gt.neighbors(i) + length(indices);
                   
                   % Add fragment ID to correspondency matrix
                   frag_corr(row_index, col_index) = j;
               else
                   % No neighbor keypoints
               end
               
           end
        end

   % end feature point loop
   end
   
end

% Generate statistics
vdisp('==============');
vdisp('Statistics:');

% Loop through fragments
for k=1:length(kp)
    vdisp('--------------');
    vdisp(['Stats for fragment ', num2str(k), ' (', kp{k}.filename, ')']);
    vdisp(['Total keypoints in fragment ', num2str(length(kp{k}.gt.x))]);
    vdisp(['Total keypoint neighbors in other fragments ', num2str(sum(sum(kp_corr(kp{k}.gt.kp_id,:))))]);
    % (kp{k}.gt.neighbors)
end

vdisp('--------------');
vdisp(['Total keypoints: ', num2str(size(kp_corr,1))]);
vdisp(['Total keypoint neighbor matches: ', num2str(sum(sum(kp_corr)))]);
vdisp(['Total keypoints with >= 1 neighbor matches: ', num2str(sum(sum(kp_corr)>0))]);
vdisp(['Total keypoints with 0 neighbor matches: ', num2str(sum(sum(kp_corr)==0))]);

% Plot correspondency matrix
if (plot_flag == 1)
    plot_corr(kp_corr,kp)
end
    
% End main function
end
   
