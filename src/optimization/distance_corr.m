function [pairs_feature, d_feature, d_xyz] = distance_corr(kp)
% Calculate correlation between feature space and ground truth distance

% Init
d_feature = [];
d_xyz = [];

% Loop through fragments
for k=1:length(kp)
    xyz1 = [kp{k}.gt.x, kp{k}.gt.y, kp{k}.gt.z];
    features1 = kp{k}.gt.features;
    
    % Init fragment xyz, features
    xyz2 = [];
    features2 = [];
    d_feature_k = [];
    d_xyz_k = [];
    pairs_feature = [];
    
    % Loop through other fragments
    for j=1:length(kp)
        if (k ~= j)
            % Collect xyz, features
            xyz2 = [xyz2; kp{j}.gt.x, kp{j}.gt.y, kp{j}.gt.z];
            features2 = [features2; kp{j}.gt.features];
        end
    end

    % For alle features1, find closes in features2
    [d_feature_k, pos] = pdist2(features2, features1, 'euclidean', 'Smallest',1);

    pairs_feature(:,1) = 1:length(pos);
    pairs_feature(:,2) = pos';

    % Get xyz distance
    for i=1:size(pairs_feature,1)
        vec1 = xyz2(pairs_feature(i,1),:);
        vec2 = xyz2(pairs_feature(i,2),:);
        d_xyz_k(i) = norm(vec1 - vec2);
    end 
    
    % Add distance to total arrays
    d_feature = [d_feature; d_feature_k'];
    d_xyz = [d_xyz; d_xyz_k'];
end


% Correlation plot
fig = figure;
plot(d_feature, d_xyz, 'b.');
title('Correlation plot (brick 1vN)');
xlabel('feature space distance');
ylabel('xyz space distance (ground truth)');
grid on;
exportgraphics(fig,'distance_corr_brick_1vN.png','Resolution',300);

