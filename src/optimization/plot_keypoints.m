function [fig_handle] = plot_keypoints(kp,kp_index1,kp_index2,offset,plot_neighbors,kp_index, kp_corr)
% Plot keypoints of two fragments in detail
% Input: offset (spacing distance between fragments)

% Handle input arguments
if nargin < 5
    plot_neighbors = 0;
end
if nargin < 4
    offset = 0;
end
disp(['offset = ', num2str(offset)]);

% Calculate offset vector
v(1) = kp{kp_index1}.gt.x_mean - kp{kp_index2}.gt.x_mean;
v(2) = kp{kp_index1}.gt.y_mean - kp{kp_index2}.gt.y_mean;
v(3) = kp{kp_index1}.gt.z_mean - kp{kp_index2}.gt.z_mean;
d = norm([v(1), v(2), v(3)]);
disp(['vx = ', num2str(v(1))]);
disp(['vy = ', num2str(v(2))]);
disp(['vz = ', num2str(v(3))]);
disp(['d = ', num2str(d)]);
o = v./d * offset;
disp(['o = ', num2str(o)]);

% Plot figure
figure;
title(['Keypoints of fragments ', num2str(kp_index1), ' & ', num2str(kp_index2)]);
hold on;
grid on;
axis equal;
hold on;

x1 = kp{kp_index1}.gt.x + o(1)/2;
y1 = kp{kp_index1}.gt.y + o(2)/2;
z1 = kp{kp_index1}.gt.z + o(3)/2;

x2 = kp{kp_index2}.gt.x - o(1)/2;
y2 = kp{kp_index2}.gt.y - o(2)/2;
z2 = kp{kp_index2}.gt.z - o(3)/2;

% Plot keypoints
plot3(x1,y1,z1,'r.');
text(kp{kp_index1}.gt.x_mean + o(1)/2,...
    kp{kp_index1}.gt.y_mean + o(2)/2,...
    kp{kp_index1}.gt.z_mean + o(3)/2,...
    num2str(kp_index1)); 

plot3(x2,y2,z2,'b.');
text(kp{kp_index2}.gt.x_mean - o(1)/2,...
    kp{kp_index2}.gt.y_mean - o(2)/2,...
    kp{kp_index2}.gt.z_mean - o(3)/2,...
    num2str(kp_index2)); 

% Plot neighbor keypoints
if plot_neighbors == 1
    % init neighbor_counter
    neighbor_counter = 0;
    
    % Get keypoints of fragment 1 with neighbors
    [I] = find(kp{kp_index1}.gt.neighbors >= 1);
    
    % Get keypoint IDs (fragment 1)
    kp_id = kp{kp_index1}.gt.kp_id(I);
    
    % Loop through keypoints with neighbors
    for i=1:length(kp_id)
        disp('-------------------');
        disp(['fragment: ', num2str(kp_index1), ' | keypoint: ', num2str(kp_id(i))]);
        disp(kp_index(kp_id(i),:));
    
        % Get neighbor keypoints (in all fragments)
        kp_neighbor = kp_corr(kp_id(i),:);
        [J] = find(kp_neighbor > 0);
        
        % Get neighbor keypoints (in fragment 2)
        kp_index_2 = kp_index(J,:);
        [K] = find(kp_index_2(:,2) == kp_index2);
        disp(['Neighbor keypoints on fragment ', num2str(kp_index2)]);
        disp(kp_index_2(K,:));
        
        % Plot line connecting neighbor keypoints
        for j=1:length(K)
            line_X = [kp{kp_index1}.gt.x(I(i))+o(1)/2, kp_index_2(K(j),3)-o(1)/2];
            line_Y = [kp{kp_index1}.gt.y(I(i))+o(2)/2, kp_index_2(K(j),4)-o(2)/2];
            line_Z = [kp{kp_index1}.gt.z(I(i))+o(3)/2, kp_index_2(K(j),5)-o(3)/2];
            line(line_X,line_Y,line_Z,'color','k','LineStyle','-');
        end % end loop j
        neighbor_counter = neighbor_counter + length(K);
      
    end % end loop i
    
    disp('=====================');
    disp(['Total keypoints fragment ',num2str(kp_index1),': ',num2str(length(kp{kp_index1}.gt.kp_id))]);
    disp(['Total keypoints fragment ',num2str(kp_index2),': ',num2str(length(kp{kp_index2}.gt.kp_id))]);
    disp(['Total keypoints with neighbors: ', num2str(neighbor_counter)]);
    disp(['= ', num2str(neighbor_counter/length(kp{kp_index1}.gt.kp_id)*100), '%']);
    
end % end if plot_neighbors

end % end main