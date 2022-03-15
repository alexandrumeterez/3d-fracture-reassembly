function []=plot_corr(kp_corr,kp)

figure;
set(gcf,'color','w');
hold on;

for k=1:size(kp_corr,1)
    for j=1:size(kp_corr,2)
        if(kp_corr(k,j)>0), plot(k,j,'b.'); end
    end
end

axis equal;
axis tight;
xlabel('keypoint ID (fragments)');
ylabel('keypoint ID (fragments)');
title('Keypoint correspndency matrix (ground truth)');

% gridlines ---------------------------
hold on
g_y(1) = 0;
g_x(1) = 0;
g_index = 1;
for i=1:length(kp)
    g_index = g_index + 1;
    g_y(g_index) = g_y(g_index-1) + length(kp{i}.gt.kp_id);
    g_x(g_index) = g_x(g_index-1) + length(kp{i}.gt.kp_id); 
end
for i=1:length(kp)
   plot([g_x(i) g_x(i)],[g_y(1) g_y(end)],'k:') %y grid lines
   plot([g_x(1) g_x(end)],[g_y(i) g_y(i)],'k:') %x grid lines
end
    

% Adjust ticks
xticks(g_x);
yticks(g_y);
%xticks(0:128:size(kp_corr,1));
%yticks(0:128:size(kp_corr,1));

% Plot identity line
plot([0,size(kp_corr,1)],[0,size(kp_corr,1)],'k-');