function [d_pairs,d_dist,gt_dist] = get_descriptor_pairs(kp, index1, index2)
% Find pairs of keypoints of two fragments with similar descriptors 
% Input: kp (keypoints struct), index fragment1, index fragment2

% Prepare output
d_pairs = [];
d_dist = [];
gt_dist = [];

% Config
max_pairs = 10;
max_d = 0.2; %Threshold value. Pairs with d>max_d will be discarded

features1 = kp{index1}.rp.features;
features2 = kp{index2}.rp.features;

% Sort keypoints by distance in feature space
if length(features2) <= length(features1)
    [D,I] = pdist2(features1, features2, 'euclidean', 'Smallest',1);
else
    [D,I] = pdist2(features2, features1, 'euclidean', 'Smallest',1);
end

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

% Calculate ground truth distance
for k=1:length(d_dist)
    A = [kp{index1}.gt.x(d_pairs(k,1)), kp{index1}.gt.y(d_pairs(k,1)), kp{index1}.gt.z(d_pairs(k,1))];
    B = [kp{index2}.gt.x(d_pairs(k,2)), kp{index2}.gt.y(d_pairs(k,2)), kp{index2}.gt.z(d_pairs(k,2))];
    gt_dist(k) = norm(A-B);
end

disp('Point 1, point 2, distance (feature space), distance (ground truth)');
disp([Array, gt_dist']);
disp('');

