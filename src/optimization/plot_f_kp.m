function [fig_handle] = plot_f_kp(f,kp,f_index)

% Plot fragment point clouds
x = f{f_index}.gt.x;
y = f{f_index}.gt.y;
z = f{f_index}.gt.z;

%figure;
hold on;
grid on;
axis equal;
set(gcf,'color','w');
plot3(x,y,z,'g.');

% Plot keypoints
x = kp{f_index}.gt.x;
y = kp{f_index}.gt.y;
z = kp{f_index}.gt.z;
plot3(x,y,z,'ro');

% Plot fragemnt number text
% text(f{f_index}.gt.x_mean, f{f_index}.gt.y_mean, f{f_index}.gt.z_mean, num2str(f_index), ...
%     'FontSize', 14, 'FontWeight', 'bold');


