function []=subplot_keypoints(subplot_id,comb_pairwise,i,A,B,A_pair,B_pair) 

% Plot random pose keypoints
%subplot(2,2,1);
%subplot(subplot_id(1),subplot_id(2),subplot_id(3));
hold on, axis equal, grid on, rotate3d on;
title(['Keypoints of fragments ',num2str(comb_pairwise(i,1)),'&',num2str(comb_pairwise(i,2))]);
plot3(A(:,1), A(:,2), A(:,3), 'b.');
plot3(A_pair(:,1), A_pair(:,2), A_pair(:,3), 'bo');
plot3(B(:,1), B(:,2), B(:,3), 'r.');
plot3(B_pair(:,1), B_pair(:,2), B_pair(:,3), 'ro');
% Plot line connecting pair keypoints
for j=1:size(A_pair,1)
    line_X = [A_pair(j,1),B_pair(j,1)];
    line_Y = [A_pair(j,2),B_pair(j,2)];
    line_Z = [A_pair(j,3),B_pair(j,3)];
    line(line_X,line_Y,line_Z,'color','k','LineStyle',':');
end % end loop j
%legend(['keypoints fragement ',num2str(comb_pairwise(i,1))],...
%    ['keypoints with neighbors fragement ',num2str(comb_pairwise(i,1))],...
%    ['keypoints fragement ',num2str(comb_pairwise(i,2))],...
%    ['keypoints with neighbors fragement ',num2str(comb_pairwise(i,2))]);
axis tight;
hold off;