function [d_pairs,d_dist,gt_dist] = get_descriptor_pairs_classical(kp, index1, index2)
% Find pairs of keypoints of two fragments with similar descriptors 
% Input: kp (keypoints struct), index fragment1, index fragment2


% Prepare output
d_pairs = [];
d_dist = [];
gt_dist = [];

% Config
max_pairs = 30;
%Threshold value. Pairs with d>max_d will be discarded
%Use 0.5 for experiments with Noise. For noise-free datasets use 0.2
max_d = 0.5;
% Parameters for Descriptor
desc_features = 3; %No of geometric features (3 or 7)
desc_scales = 5; %(No of radii (multi-scale))
% Use 3 and 5 for all sample datasets provided or set them according to the
% params used in keypoint extraction

desc_mat_sz = desc_features*desc_features;

X = kp{index1}.gt.features;
Y = kp{index2}.gt.features;

%Multi-scale average of Frobenius Norm

k = 1;
D = pdist2(Y(:,(k-1)*desc_mat_sz +1:k*desc_mat_sz), X(:,(k-1)*desc_mat_sz +1:k*desc_mat_sz), 'euclidean');

for k=2:desc_scales
    D = D + pdist2(Y(:,(k-1)*desc_mat_sz +1:k*desc_mat_sz), X(:,(k-1)*desc_mat_sz +1:k*desc_mat_sz), 'euclidean');
end
    D = D/desc_scales;
    [D, I] = min(D, [], 1);

% Create array with point ID1, ID2, distance
Array(:,1) = 1:length(I);
Array(:,2) = I';
Array(:,3) = D;

% Sort by distance ascending
Array = sortrows(Array,3);

% Cut off after max_pairs
Array = Array(1:max_pairs,:);

% Filter out d > max_d
filter = Array(:,3) <= max_d;
Array = Array(filter,:);

% Prepare output
d_pairs = Array(:,1:2);
d_dist = Array(:,3);
gt_dist = zeros(size(Array,1),1)';

% Calculate ground truth distance
for k=1:length(d_dist)
    A = [kp{index1}.gt.x(d_pairs(k,1)), kp{index1}.gt.y(d_pairs(k,1)), kp{index1}.gt.z(d_pairs(k,1))];
    B = [kp{index2}.gt.x(d_pairs(k,2)), kp{index2}.gt.y(d_pairs(k,2)), kp{index2}.gt.z(d_pairs(k,2))];
    gt_dist(k) = norm(A-B);
end

disp('Point 1, point 2, distance (feature space), distance (ground truth)');
disp([Array, gt_dist']);
disp('');

