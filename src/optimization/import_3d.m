function [f,kp,ft] = import_3d
% Descripton: Import fragment, keypoints 3D clouds, feature descriptors
% Output: f (fragments), kp (keypoints), feature descriptors (ft)
% -------------------------------
% Semester project 3D vision lecture
% 3D Feature Point Learning for Fractured Object Reassembly
% Team: Du Chaoyu (chaoyu.du@arch.ethz.ch)
% Ulrich Steger (ulsteger@student.ethz.ch)
% Eshaan Mudga (emudgal@student.ethz.ch) 
% Florian Hürlimann (fhuerliman@student.ethz.ch)
% Author of this code: F. Huerlimann

% nomenclature:
% gt = ground truth
% rp = random pose
% R = rotation matrix (3x3)
% T = translation matrix (3x1)

% Config
% ------
% Cube_6 dataset (synthetic/blender)
data_dir_fragment = 'data/fragments/cube_6'; % fragment *.npy directory
data_dir_kp = 'data/keypoints/cube_6_FN'; % keypoint *.npy directory
data_dir_ft = 'data/encoded_desc/cube_6_FN'; % features *.npy directory

% Brick dataset (3d scan)
%data_dir_fragment = 'data/fragments/brick'; % fragment *.npy directory
%data_dir_kp = 'data/keypoints/brick_1vN'; % keypoint *.npy directory
%data_dir_ft = 'data/encoded_desc/brick_1vN'; % features *.npy directory

rng(0); % Initialize random seed for reproducibility
max_rotation = pi; % Max angle (rad) for random pose
max_translation = 1; % Max translation (-) for random pose

verbose_flag = 1;
plot_flag = 1;

% Verbose output function
function vdisp(mystring)
    if verbose_flag == true
        disp(mystring); 
    end
end

% Input data definition
% ---------------------
% Get filenames of fragment files
old_dir = pwd;
new_dir = fullfile(old_dir, data_dir_fragment);
cd(new_dir);
d = dir('*.npy');
for k=1:length(d)
    f{k}.filename = d(k).name;
end
cd(old_dir);

% Get filenames of keypoint files
new_dir = fullfile(old_dir, data_dir_kp);
cd(new_dir);
d = dir('*.npy');
for k=1:length(d)
    kp{k}.filename = d(k).name;
end
cd(old_dir);

% Get filenames of features files
new_dir = fullfile(old_dir, data_dir_ft);
cd(new_dir);
d = dir('*.npy');
for k=1:length(d)
    ft{k}.filename = d(k).name;
end
cd(old_dir);

if length(f)~=length(kp) || length(f)~=length(ft)
    error('unequal number of fragments.');
else
    disp(['Number of fragments: ',num2str(length(f))]);
end
    
    
% Import 3D fragments
% -------------------
disp('---');
disp('Importing fragments ...');
disp(['Directory: ',data_dir_fragment]);
for k=1:length(f)
    disp(['Import ', fullfile(data_dir_fragment, f{k}.filename)]);
    npy = readNPY(fullfile(data_dir_fragment, f{k}.filename));
    
    % Store point coordinate, normals in ground truth (gt) struct
    f{k}.gt.x = npy(:,1);
    f{k}.gt.y = npy(:,2);
    f{k}.gt.z = npy(:,3);
    f{k}.gt.n1 = npy(:,4);
    f{k}.gt.n2 = npy(:,5);
    f{k}.gt.n3 = npy(:,6);
    
    % Calculate xyz dimension
    f{k}.gt.x_size = max(npy(:,1)) - min(npy(:,1));
    f{k}.gt.y_size = max(npy(:,2)) - min(npy(:,2));
    f{k}.gt.z_size = max(npy(:,3)) - min(npy(:,3));
    
    % Calculate mean point
    f{k}.gt.x_mean = mean(npy(:,1));
    f{k}.gt.y_mean = mean(npy(:,2));
    f{k}.gt.z_mean = mean(npy(:,3));
    clear npy;
end

% Import 3D keypoints
% -------------------
disp('---');
disp('Importing keypoints ...');
disp(['Directory: ',data_dir_kp]);
for k=1:length(kp)
    disp(['Import ', fullfile(data_dir_kp, kp{k}.filename)]);
    npy = readNPY(fullfile(data_dir_kp, kp{k}.filename));
    
    % Store point coordinate, sigma (saliency) in ground truth (gt) struct
    kp{k}.gt.x = npy(:,1);
    kp{k}.gt.y = npy(:,2);
    kp{k}.gt.z = npy(:,3);
    kp{k}.gt.sigma = npy(:,4);
    
    % Calculate xyz dimension
    kp{k}.gt.x_size = max(npy(:,1)) - min(npy(:,1));
    kp{k}.gt.y_size = max(npy(:,2)) - min(npy(:,2));
    kp{k}.gt.z_size = max(npy(:,3)) - min(npy(:,3));
    
    % Calculate mean point
    kp{k}.gt.x_mean = mean(npy(:,1));
    kp{k}.gt.y_mean = mean(npy(:,2));
    kp{k}.gt.z_mean = mean(npy(:,3));
    
    % Init number of neighbors
    kp{k}.gt.neighbors = zeros(size(kp{k}.gt.x));
    clear npy;
end

% Import 3D features
% ------------------
disp('---');
disp('Importing features ...');
disp(['Directory: ',data_dir_ft]);
for k=1:length(ft)
    disp(['Import ', fullfile(data_dir_ft, ft{k}.filename)]);
    npy = readNPY(fullfile(data_dir_ft, ft{k}.filename));
    
    if size(npy,1) == length(kp{k}.gt.x)
        kp{k}.gt.features = npy;
    else
        error('Number of keypoints does not match number of features.');
    end
end


% Create random pose fragments, keypoints
% ---------------------------------------
disp('---');
% Loop through fragments
for k=1:length(f)
    vdisp(['Create random pose for fragment: ', num2str(k)]);
    rx = rand(1)*max_rotation;
    ry = rand(1)*max_rotation;
    rz = rand(1)*max_rotation;
    tx = rand(1)*max_translation;
    ty = rand(1)*max_translation;
    tz = rand(1)*max_translation;
    T = [tx; ty; tz];
    f{k}.rp.T = T;
    
    Rx = [1 0 0; 0 cos(rx) -sin(rx); 0 sin(rx) cos(rx)];
    Ry = [cos(ry) 0 sin(ry); 0 1 0; -sin(ry) 0 cos(ry)];
    Rz = [cos(rz) -sin(rz) 0; sin(rz) cos(rz) 0; 0 0 1];
    R = Rx*Ry*Rz;
    f{k}.rp.R = R;
    
    f{k}.rp.transformation_rp = [R, T; [0 0 0 1]];
    
    % Apply transformation to ground truth point cloud (gt)
    xyz_gt = [f{k}.gt.x, f{k}.gt.y, f{k}.gt.z];
    xyz_rp = (R*xyz_gt')';
    
    % Random pose (rp)
    f{k}.rp.x = xyz_rp(:,1) + tx;
    f{k}.rp.y = xyz_rp(:,2) + ty;
    f{k}.rp.z = xyz_rp(:,3) + tz;
    clear xyz_gt xyz_rp;
    
    % Calculate mean point
    f{k}.rp.x_mean = mean(f{k}.rp.x);
    f{k}.rp.y_mean = mean(f{k}.rp.y);
    f{k}.rp.z_mean = mean(f{k}.rp.z);
    
    % Transform keypoints
    xyz_gt = [kp{k}.gt.x, kp{k}.gt.y, kp{k}.gt.z];
    xyz_rp = (R*xyz_gt')';
    kp{k}.rp.x = xyz_rp(:,1) + tx;
    kp{k}.rp.y = xyz_rp(:,2) + ty;
    kp{k}.rp.z = xyz_rp(:,3) + tz;
    clear xyz_gt xyz_rp;
    kp{k}.rp.T = T;
    kp{k}.rp.R = R;
    kp{k}.rp.transformation_rp = [R, T; [0 0 0 1]];
    
    % Add feature descriptor sigma (saliency score)
    kp{k}.rp.sigma = kp{k}.gt.sigma;
    
    % Calculate mean point
    kp{k}.rp.x_mean = mean(kp{k}.rp.x);
    kp{k}.rp.y_mean = mean(kp{k}.rp.y);
    kp{k}.rp.z_mean = mean(kp{k}.rp.z);
    
    % Add features
    kp{k}.rp.features = kp{k}.gt.features;
end


% Plot point clouds, keypoints
% ----------------------------
if plot_flag == true
    fig.f = figure;
    set(gcf,'color','w');
    fig.sub1 = subplot(1,2,1);
    hold on;
    axis equal;
    grid on;
    title('Fragments (ground truth)');
    for k=1:length(f)
        plot3(f{k}.gt.x, f{k}.gt.y, f{k}.gt.z, '.');
        
        % Plot fragemnt number text
        text(f{k}.gt.x_mean, f{k}.gt.y_mean, f{k}.gt.z_mean, num2str(k), ...
            'FontSize', 14, 'FontWeight', 'bold');
    end
    hold off;
    
    %figure;
    fig.sub2 = subplot(1,2,2);
    hold on;
    axis equal;
    grid on;
    title('Keypoints (ground truth)');
    for k=1:length(kp)
        plot3(kp{k}.gt.x, kp{k}.gt.y, kp{k}.gt.z, '.');
        
        % Plot fragemnt number text
        text(kp{k}.gt.x_mean, kp{k}.gt.y_mean, kp{k}.gt.z_mean, num2str(k), ...
            'FontSize', 14, 'FontWeight', 'bold');
    end

    figure;
    set(gcf,'color','w');
    fig.sub3 = subplot(1,2,1);
    hold on;
    axis equal;
    axis tight;
    grid on;
    title('Keypoints (random pose)');
    for k=1:length(f)
        plot3(f{k}.rp.x, f{k}.rp.y, f{k}.rp.z, '.');
        
        % Plot fragemnt number text
        text(f{k}.rp.x_mean, f{k}.rp.y_mean, f{k}.rp.z_mean, num2str(k), ...
            'FontSize', 14, 'FontWeight', 'bold');
    end
    hold off;
    
    %figure;
    fig.sub4 = subplot(1,2,2);
    hold on;
    axis equal;
    axis tight;
    grid on;
    title('Keypoints (random pose)');
    for k=1:length(kp)
        plot3(kp{k}.rp.x, kp{k}.rp.y, kp{k}.rp.z, '.');
        
        % Plot fragemnt number text
        text(kp{k}.rp.x_mean, kp{k}.rp.y_mean, kp{k}.rp.z_mean, num2str(k), ...
            'FontSize', 14, 'FontWeight', 'bold');
    end
    hold off;
end

%fig = gcf; exportgraphics(fig,'distance_correlation.png','Resolution',300);
% End main function
end
