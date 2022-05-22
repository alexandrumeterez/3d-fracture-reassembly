%%% Plot all pointclouds (*.pcd) located in the same folder

files = dir("*.pcd");

figure()
hold on
v = 0;
for k = 1:length(files)
    name = files(k).name;
%     clc
    disp("Reading File: " + name)
    s = importdata(name,' ',1).data;
    v = v + size(s,1);
    % plot pointcloud with sureface normal as color
    pcshow(s(:,1:3), s(:,4:6))
end
disp("Points Count: " + v)
