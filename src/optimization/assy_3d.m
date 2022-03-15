function [f_out, kp_out] = assy_3d(f,kp,kp_corr,frag_corr)
% Description: perform 3D fragment matching and reassembly
% Based on "main_3d_voronoi.m" (GitHub: FragRe-master) by Pablo Speciale
% (pablo.speciale@microsoft.com) and Cui Luxiao
% Input: f (fragments), kp (keypoints), kp_corr (keypoint correlation
% matrix), frag_corr (fragment correlation matrix)
% Output: matching_pairwise (struct)
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
plot_poses_flag = 0; % Plot all pairwise poses
plot_result_flag = 1; % Plot graph, assembly
use_ground_truth = 1; % keypoint pair selection based on ground truth
use_rigid_transformation = 1; % enable helmert_nd()
add_isolated_fragments = 1; % fop fragments without triple match

% Pairwise matching config
% Min allowable scaling factor for pose estimation
S_bnb_min = 0.7; 

% Triplewise matching config
% Set thresholds for match (Rotation angle (RAD), Transformation)
th_R = 1.0; th_T = 10;

% Check number of fragments (fragments vs. keypoints)
num_of_fragments = length(f);
if length(f)~=length(kp)
    error('unequal number of fragments.');
end

% Generate fragment combinations
comb_pairwise = nchoosek(1:num_of_fragments,2);
matching_pairwise = cell(size(comb_pairwise,1),1);
nb_non_matches = 0;

% Init pairwise matching array (0 = no match, 1 = match)
match_pairwise = zeros(size(comb_pairwise,1),1);


%% Loop through fragment combinations (loop 1)
% -------------------------------------------
for i = 1:size(comb_pairwise,1)
    disp(['Combination ', num2str(i), '/', num2str(size(comb_pairwise,1)), ' | Loop 1']); 
    disp(['Fragments (A,B): ', num2str(comb_pairwise(i,1)), ', ', num2str(comb_pairwise(i,2))]);
    
    % Get random poise (rp), ground truth (gt) keypoints of fragment A, B
    A_rp = [kp{comb_pairwise(i,1)}.rp.x, kp{comb_pairwise(i,1)}.rp.y, kp{comb_pairwise(i,1)}.rp.z];
    A_gt = [kp{comb_pairwise(i,1)}.gt.x, kp{comb_pairwise(i,1)}.gt.y, kp{comb_pairwise(i,1)}.gt.z];
    
    B_rp = [kp{comb_pairwise(i,2)}.rp.x, kp{comb_pairwise(i,2)}.rp.y, kp{comb_pairwise(i,2)}.rp.z];
    B_gt = [kp{comb_pairwise(i,2)}.gt.x, kp{comb_pairwise(i,2)}.gt.y, kp{comb_pairwise(i,2)}.gt.z];
    
    % Init corresponding keypoint pairs
    A_gt_pair = [];
    A_rp_pair = [];
    B_gt_pair = [];
    B_rp_pair = [];
    pair_nb = 0;
    
    if use_ground_truth==true
        % Loop through keypoints of A and find neighbors in B
        A_kp = kp{comb_pairwise(i,1)}.gt.kp_id;
        for k=1:length(A_kp)
           [I,J] = find(kp_corr(A_kp(k),:) > 0);
           if I > 0
            if frag_corr(A_kp(k),J) == comb_pairwise(i,2)
               % Match!
               pair_nb = pair_nb + 1;
               A_rp_pair(pair_nb,:) = [kp{comb_pairwise(i,1)}.rp.x(k), kp{comb_pairwise(i,1)}.rp.y(k), kp{comb_pairwise(i,1)}.rp.z(k)];
               A_gt_pair(pair_nb,:) = [kp{comb_pairwise(i,1)}.gt.x(k), kp{comb_pairwise(i,1)}.gt.y(k), kp{comb_pairwise(i,1)}.gt.z(k)];

               [K,L] = find(kp{comb_pairwise(i,2)}.rp.kp_id == J(1));
               B_rp_pair(pair_nb,:) = [kp{comb_pairwise(i,2)}.rp.x(L), kp{comb_pairwise(i,2)}.rp.y(L), kp{comb_pairwise(i,2)}.rp.z(L)];
               B_gt_pair(pair_nb,:) = [kp{comb_pairwise(i,2)}.gt.x(L), kp{comb_pairwise(i,2)}.gt.y(L), kp{comb_pairwise(i,2)}.gt.z(L)];
            end
           end
        end % end loop
    else
        %Find keypoint pairs according to feature space distance (May 28th)
        [d_pairs,d_dist,gt_dist] = get_descriptor_pairs(kp, comb_pairwise(i,1), comb_pairwise(i,2));
        A_rp_pair = [kp{comb_pairwise(i,1)}.rp.x(d_pairs(:,1)), kp{comb_pairwise(i,1)}.rp.y(d_pairs(:,1)), kp{comb_pairwise(i,1)}.rp.z(d_pairs(:,1))];
        A_gt_pair = [kp{comb_pairwise(i,1)}.gt.x(d_pairs(:,1)), kp{comb_pairwise(i,1)}.gt.y(d_pairs(:,1)), kp{comb_pairwise(i,1)}.gt.z(d_pairs(:,1))];
        B_rp_pair = [kp{comb_pairwise(i,2)}.rp.x(d_pairs(:,2)), kp{comb_pairwise(i,2)}.rp.y(d_pairs(:,2)), kp{comb_pairwise(i,2)}.rp.z(d_pairs(:,2))];
        B_gt_pair = [kp{comb_pairwise(i,2)}.gt.x(d_pairs(:,2)), kp{comb_pairwise(i,2)}.gt.y(d_pairs(:,2)), kp{comb_pairwise(i,2)}.gt.z(d_pairs(:,2))];
    end
    
    disp(['Number of keypoint pairs: ',num2str(length(A_rp_pair))]);

    % Prepare input data for solver
    ptsA = A_rp_pair';
    ptsB = B_rp_pair';
    zcA = zeros(1,size(ptsA,2)); 
    ptsA_z = double([ptsA;zcA]);
    zcB = zeros(1,size(ptsB,2)); 
    ptsB_z = double([ptsB;zcB]);

    % Solve numerical optimization with YALMIP/Mosek
    % (https://yalmip.github.io/download/)
    % Transformation (R, T) to map pts1 to pts2 (pts1_transformed = R_bnb*ptsA+T_bnb)
    [R_bnb, T_bnb, S_bnb, inliers_bnb, x_opt, sol_yalmip] = ...
        computeRTSwithSDP_yalmip_bnb_3d(ptsA_z, ptsB_z);
            
    % Check validity of solution, save and apply transformation matrix
    if ~isnan(T_bnb(1,1)) && ~(sum(sum(T_bnb==zeros(3)))==9) && S_bnb >= S_bnb_min
        disp('Valid solution for T_bnb');
        disp(['S_bnb = ', num2str(S_bnb)]); 
        
        % Get rigid transformation
        if use_rigid_transformation == 1
            [S_bnb, R_bnb, T_bnb] = helmert_nd(ptsA(:,inliers_bnb), ptsB(:,inliers_bnb), S_bnb, R_bnb, T_bnb);
        end
        
        matching_pairwise{i}.transformation_A = [R_bnb T_bnb; 0 0 0 1]; 
        nb_non_matches = nb_non_matches + 1;
        
        % Apply transformation (R,T) to fragment A
        ptsA_transformed = R_bnb * A_rp_pair' + T_bnb;
        ptsA_all_transformed = R_bnb * A_rp' + T_bnb;
        
        % Update match pairwise array
        match_pairwise(i) = 1;
        
        % Check affinity of rotation matrix
        check_R(R_bnb);
        
    else
        disp('No valid solution for T_bnb -> create NaN R, T');
        R_bnb = NaN(3,3);
        T_bnb = NaN(3,1);
        %disp('No valid solution for T_bnb -> create random T');
        %T_bnb = 1000*rand(3,1);
        matching_pairwise{i}.transformation_A = [R_bnb T_bnb; 0 0 0 1]; 
        nb_non_matches = nb_non_matches + 1;
    end
        
    % Plot random pose keypoints
    %if (plot_poses_flag == 1 && ~isnan(T_bnb(1,1)))
    if plot_poses_flag == 1
        fig.f = figure;
        set(gcf,'color','w');
        
        fig.sub1 = subplot(1,3,1);
        subplot_keypoints([1,3,1],comb_pairwise,i,A_gt,B_gt,A_gt_pair,B_gt_pair); 
        title('Ground truth pose');
        
        fig.sub2 = subplot(1,3,2);
        subplot_keypoints([1,3,2],comb_pairwise,i,A_rp,B_rp,A_rp_pair,B_rp_pair); 
        title('Random pose');
        
        % Plot only if pose estimation successful
        if ~isnan(T_bnb(1,1))
            fig.sub3 = subplot(1,3,3);
            subplot_keypoints([1,3,3],comb_pairwise,i,ptsA_all_transformed',B_rp,ptsA_transformed',B_rp_pair);
            title('Random pose transformed');
        end
        
        %camorbit_animation(fig,'pose_estimation');
        
        %disp(['Number of keypoint pairs: ',num2str(length(d_pairs))]);

        fig.corr = figure;
        plot(d_dist,gt_dist,'.');
        xlabel('feature space distance');
        ylabel('ground truth distance');
        close(fig.corr);
        close(fig.f);
    end
end % end loop


%% Loop through fragment combinations (loop 2)
% -------------------------------------------
for i = 1:size(comb_pairwise,1)
disp(['Combination ', num2str(i), '/', num2str(size(comb_pairwise,1)), ' | Loop 2']); 
    disp(['Fragments (A,B): ', num2str(comb_pairwise(i,2)), ', ', num2str(comb_pairwise(i,1))]);
    
    % Get random poise (rp), ground truth (gt) keypoints of fragment A, B
    A_rp = [kp{comb_pairwise(i,2)}.rp.x, kp{comb_pairwise(i,2)}.rp.y, kp{comb_pairwise(i,2)}.rp.z];
    A_gt = [kp{comb_pairwise(i,2)}.gt.x, kp{comb_pairwise(i,2)}.gt.y, kp{comb_pairwise(i,2)}.gt.z];
    B_rp = [kp{comb_pairwise(i,1)}.rp.x, kp{comb_pairwise(i,1)}.rp.y, kp{comb_pairwise(i,1)}.rp.z];
    B_gt = [kp{comb_pairwise(i,1)}.gt.x, kp{comb_pairwise(i,1)}.gt.y, kp{comb_pairwise(i,1)}.gt.z];
    
    % Init corresponding keypoint pairs
    A_gt_pair = [];
    A_rp_pair = [];
    B_gt_pair = [];
    B_rp_pair = [];
    pair_nb = 0;
    
    if use_ground_truth==true
        % Loop through keypoints of A and find neighbors in B
        A_kp = kp{comb_pairwise(i,2)}.gt.kp_id;
        for k=1:length(A_kp)
           [I,J] = find(kp_corr(A_kp(k),:) > 0);
           if I > 0
            if frag_corr(A_kp(k),J) == comb_pairwise(i,1)
               % Match!
               pair_nb = pair_nb + 1;
               A_rp_pair(pair_nb,:) = [kp{comb_pairwise(i,2)}.rp.x(k), kp{comb_pairwise(i,2)}.rp.y(k), kp{comb_pairwise(i,2)}.rp.z(k)];
               A_gt_pair(pair_nb,:) = [kp{comb_pairwise(i,2)}.gt.x(k), kp{comb_pairwise(i,2)}.gt.y(k), kp{comb_pairwise(i,2)}.gt.z(k)];

               [K,L] = find(kp{comb_pairwise(i,1)}.rp.kp_id == J(1));
               B_rp_pair(pair_nb,:) = [kp{comb_pairwise(i,1)}.rp.x(L), kp{comb_pairwise(i,1)}.rp.y(L), kp{comb_pairwise(i,1)}.rp.z(L)];
               B_gt_pair(pair_nb,:) = [kp{comb_pairwise(i,1)}.gt.x(L), kp{comb_pairwise(i,1)}.gt.y(L), kp{comb_pairwise(i,1)}.gt.z(L)];
            end
           end
        end
    else
        % Find keypoint pairs according to feature space distance (May 28th)
        [d_pairs,d_dist] = get_descriptor_pairs(kp, comb_pairwise(i,1), comb_pairwise(i,2));
        A_rp_pair = [kp{comb_pairwise(i,2)}.rp.x(d_pairs(:,2)), kp{comb_pairwise(i,2)}.rp.y(d_pairs(:,2)), kp{comb_pairwise(i,2)}.rp.z(d_pairs(:,2))];
        A_gt_pair = [kp{comb_pairwise(i,2)}.gt.x(d_pairs(:,2)), kp{comb_pairwise(i,2)}.gt.y(d_pairs(:,2)), kp{comb_pairwise(i,2)}.gt.z(d_pairs(:,2))];
        B_rp_pair = [kp{comb_pairwise(i,1)}.rp.x(d_pairs(:,1)), kp{comb_pairwise(i,1)}.rp.y(d_pairs(:,1)), kp{comb_pairwise(i,1)}.rp.z(d_pairs(:,1))];
        B_gt_pair = [kp{comb_pairwise(i,1)}.gt.x(d_pairs(:,1)), kp{comb_pairwise(i,1)}.gt.y(d_pairs(:,1)), kp{comb_pairwise(i,1)}.gt.z(d_pairs(:,1))];
    end

    disp(['Number of keypoint pairs: ',num2str(length(A_rp_pair))]);

    % Prepare input data for solver
    ptsA = A_rp_pair';
    ptsB = B_rp_pair';
    zcA = zeros(1,size(ptsA,2)); 
    ptsA_z = double([ptsA;zcA]);
    zcB = zeros(1,size(ptsB,2)); 
    ptsB_z = double([ptsB;zcB]);

    % Solve numerical optimization with YALMIP/Mosek
    % (https://yalmip.github.io/download/)
    % Transformation (R, T) to map pts1 to pts2 (pts1_transformed = R_bnb*ptsA+T_bnb)
    [R_bnb, T_bnb, S_bnb, inliers_bnb, x_opt, sol_yalmip] = ...
        computeRTSwithSDP_yalmip_bnb_3d(ptsA_z, ptsB_z);
        
    % Check validity of solution, save and apply transformation matrix
    %if ~isnan(T_bnb(1,1)) && ~(sum(sum(T_bnb==zeros(3)))==9) && (abs(S_bnb-1.0) <= 0.3)
    if ~isnan(T_bnb(1,1)) && ~(sum(sum(T_bnb==zeros(3)))==9) && S_bnb >= S_bnb_min
        disp('Valid solution for T_bnb');
        disp(['S_bnb = ', num2str(S_bnb)]); 
        
        % Get rigid transformation
        if use_rigid_transformation == 1
            [S_bnb, R_bnb, T_bnb] = helmert_nd(ptsA(:,inliers_bnb), ptsB(:,inliers_bnb), S_bnb, R_bnb, T_bnb);
        end
        
        matching_pairwise{i}.transformation_B = [R_bnb T_bnb; 0 0 0 1]; 
        nb_non_matches = nb_non_matches + 1;
        
        % Apply transformation (R,T) to fragment A
        ptsA_transformed = R_bnb * A_rp_pair' + T_bnb;
        ptsA_all_transformed = R_bnb * A_rp' + T_bnb;    
        
        % Update match pairwise array
        match_pairwise(i) = 1;
        
        % Check affinity of rotation matrix
        check_R(R_bnb);
    else
        disp('No valid solution for T_bnb -> create NaN R, T');
        R_bnb = NaN(3,3);
        T_bnb = NaN(3,1);
        %disp('No valid solution for T_bnb -> create random T');
        %T_bnb = 1000*rand(3,1);
        matching_pairwise{i}.transformation_B = [R_bnb T_bnb; 0 0 0 1]; 
        nb_non_matches = nb_non_matches + 1;
    end
        
    % Plot random pose keypoints
    %if (plot_poses_flag == 1 && ~isnan(T_bnb(1,1)))
    if plot_poses_flag == 1
        fig.f = figure;
        set(gcf,'color','w');
        
        fig.sub1 = subplot(1,3,1);
        subplot_keypoints([1,3,1],comb_pairwise,i,A_gt,B_gt,A_gt_pair,B_gt_pair); 
        title('Ground truth pose');
        
        fig.sub2 = subplot(1,3,2);
        subplot_keypoints([1,3,2],comb_pairwise,i,A_rp,B_rp,A_rp_pair,B_rp_pair); 
        title('Random pose');
        
        % Plot only if pose estimation successful
        if ~isnan(T_bnb(1,1))
            fig.sub3 = subplot(1,3,3);
            subplot_keypoints([1,3,3],comb_pairwise,i,ptsA_all_transformed',B_rp,ptsA_transformed',B_rp_pair);
            title('Random pose transformed');
        end
        
        %camorbit_animation(fig,'pose_estimation');
        
        fig.corr = figure;
        plot(d_dist,gt_dist,'.');
        xlabel('feature space distance');
        ylabel('ground truth distance');
        close(fig.corr);
        close(fig.f)
    end
end


%% Plot results of pairwise matching
disp('----');
disp('[comb_pairwise, match_pairwise]');
disp([comb_pairwise, match_pairwise]);


%% Verify that pairwise match results in 2 transformation matrices A, B
% (e.g. frag2 = A*frag1, frag1 = B*frag2)
% If one is missing, substitute with inverse transformation matrix of other
disp('Checking for missing transformation matrices');

% Loop through pairwise combinations
for k=1:length(match_pairwise)
    if match_pairwise(k) == 1 % match
        disp('------');
        disp(['match_pairwise(',num2str(k),')']);
        A_isnan = sum(sum(isnan(matching_pairwise{k}.transformation_A)));
        B_isnan = sum(sum(isnan(matching_pairwise{k}.transformation_B)));
        
        if (A_isnan == 0 && B_isnan == 0) % A, B OK (not NaN)
            disp('A, B OK (not NaN)');
        end
        
        if (A_isnan == 0 && B_isnan > 0) % B is missing
            disp('B is emtpy. Replacing with inverse of A');
            matching_pairwise{k}.transformation_B = matching_pairwise{k}.transformation_A^-1;
        end
        
        if (A_isnan > 0 && B_isnan == 0) % A is missing
             disp('A is emtpy. Replacing with inverse of B');
            matching_pairwise{k}.transformation_A = matching_pairwise{k}.transformation_B^-1;
        end
       
        % Print matrices
        disp('A:');
        disp(matching_pairwise{k}.transformation_A);
        disp('B:');
        disp(matching_pairwise{k}.transformation_B);       
    end  
end



%% Triple-wise matching
comb_triplewise = nchoosek(1:num_of_fragments,3);
matching_triplewise = cell(size(comb_triplewise,1),1);
constraint_triple = zeros(size(comb_triplewise,1),1);

% Init pairwise matching array (0 = no match, 1 = match)
match_triplewise = zeros(size(comb_pairwise,1),1);

% Loop through fragment triples
for i = 1:size(comb_triplewise,1)
    [T_gt,T_est,constraint] = triplewise_matching_3d(comb_pairwise,matching_pairwise,comb_triplewise(i,:),th_R,th_T);
    matching_triplewise{i}.T_gt = T_gt;
    matching_triplewise{i}.T_est = T_est;
    constraint_triple(i) = constraint;
    
    if constraint == 0 % (= tripleweise match)
       
        % Update triplewise match array (0 = no match, 1 = match)
        triple_index = comb_triplewise(i,:);
        disp(['Triplewise match: ', num2str(triple_index)]);
        index_12 = triple_index(:,1:2);
        index_13 = [triple_index(:,1) triple_index(:,3)];
        index_23 = triple_index(:,2:3);
        
        % Find location of comb_pairwise in index_**
        [~,loc_12] = ismember(index_12,comb_pairwise,'rows');
        [~,loc_13] = ismember(index_13,comb_pairwise,'rows');
        [~,loc_23] = ismember(index_23,comb_pairwise,'rows');
        
        match_triplewise(loc_12) = 1;
        match_triplewise(loc_13) = 1;
        match_triplewise(loc_23) = 1;
    end
end
disp('Triple matching constraints:');
disp(['Number of triplewise matching constraints: ', num2str(size(comb_triplewise,1))]);
disp(['Number of triplewise matching constraints == 0: ', num2str(size(comb_triplewise,1)-sum(sum(constraint_triple)))]);
disp(['Number of triplewise matching constraints == 1: ', num2str(sum(sum(constraint_triple)))]);


%% Discovery of spatial adjacent fragments
adjacency_pairs = [];
for i=1:size(comb_pairwise,1)
   if (match_triplewise(i) == 1)
        adjacency_pairs = [adjacency_pairs; comb_pairwise(i,:)]; 
   end
end
disp(['number of adjacency pairs: ', num2str(size(adjacency_pairs,1))]);


%% Overall Reassembly
count_neighbor = zeros(num_of_fragments,1);
for i = 1:size(adjacency_pairs,1)
    index1 = adjacency_pairs(i,1);
    index2 = adjacency_pairs(i,2);
    count_neighbor(index1) = count_neighbor(index1) + 1;
    count_neighbor(index2) = count_neighbor(index2) + 1;
end

% Construct graph
s = adjacency_pairs(:,1)';
t = adjacency_pairs(:,2)';
G = graph(s,t);


%% Add lost fragements (nodes)
% (= graph nodes with no edge, e.g. because no triple-match)
if add_isolated_fragments == 1

    % Get number of edges per graph node
    D = degree(G);
    D_has_edge = find(D>0);
    D_no_edge = find(D==0);

    % Loop through nodes without edges
    disp('--------');
    disp('Adding lost fragments ...');
    for i=1:length(D_no_edge)
        this_node = D_no_edge(i);
        disp(['Node without edges: ', num2str(this_node)]);

        % Init new coordinatates
        f{this_node}.assy = f{this_node}.rp;
        kp{this_node}.assy = kp{this_node}.rp;

        % Loop through all nodes of graph
        for k=1:numnodes(G)
            node2 = k;
            [~,position] = ismember([this_node,node2],comb_pairwise,'rows');
            if position ~= 0 && match_pairwise(position)==1 % pairwise match this_node<->node2
                disp(['Pairwise match with node: ', num2str(node2)]);
                disp(['Adding edge ',num2str(this_node),' -> ',num2str(node2)]);
                G = addedge(G,this_node,node2);
            end                

            [~,position] = ismember([node2,this_node],comb_pairwise,'rows');
            if position ~= 0 && match_pairwise(position)==1 % pairwise match node2<->this_node
                disp(['Pairwise match with node: ', num2str(node2)]);
                disp(['Adding edge ',num2str(this_node),' -> ',num2str(node2)]);
                G = addedge(G,this_node,node2);
            end    
        end
    end % end loop
    
    % Simplify graph (remove redundant edges)
    G = simplify(G);
end % end if


%% Plot graph
figure(7); h = plot(G,'MarkerSize',8); h.NodeColor = 'red';
set(gcf,'color','w');
title('Fragment adjacency graph');
nl = h.NodeLabel;
h.NodeLabel = '';
xd = get(h, 'XData')+0.06;
yd = get(h, 'YData');
text(xd, yd, nl,'FontSize',12,'FontWeight','bold', 'HorizontalAlignment','left');

%% Init new (reassembled) fragments
for i=1:length(f)
   f{i}.assy = [];
   kp{i}.assy = [];
end

% Define reference (start) node
ref_node = 6; % Manually chosen
disp(['Reference (start) node = ',num2str(ref_node)]);
f{ref_node}.assy = f{ref_node}.rp;
kp{ref_node}.assy = kp{ref_node}.rp;

% Traverse the graph from the reference node
v = bfsearch(G, ref_node); % Get nodes connected to ref_node

% Loop through connected nodes
for i = 2:size(v,1)
    % note: v(1) = ref_node
    if v(1) ~= ref_node
        error('ref_node NOK');
    end
    
    % Select node i
    this_node = v(i);
    disp('------');
    disp(['this_node i(',num2str(i),') = ',num2str(v(i))]);
    
    % Copy coordinates from random pose i
    f{this_node}.assy = f{this_node}.rp;
    kp{this_node}.assy = kp{this_node}.rp;
    
    % Get path from ref_node to this_node
    path = shortestpath(G, ref_node, this_node);
    disp(['Shortest path from ref_node to this_node: ',num2str(path)]);
    
    % Init total transformation matrix
    transformation_temp = [];
    
    % Loop through path
    %for k=1:(length(path)-1)
    for k=(length(path)-1):-1:1
        disp(['Path segment ', num2str(k)]);
        
        pair_index = sort(path(k:k+1));
        disp(['pair_index: ', num2str(pair_index)]);
        
        [~,position] = ismember(pair_index,comb_pairwise,'rows');
        if (path(k) == pair_index(1))
            % Original order
            
            % Get transformation matrix
            transf_est = matching_pairwise{position}.transformation_B;
            
            % Transform fragment point cloud
            xyz_old = [f{this_node}.assy.x, f{this_node}.assy.y, f{this_node}.assy.z];
            xyz_old(:,4) = 1;
            xyz_new = (transf_est * xyz_old')';
            f{this_node}.assy.x = xyz_new(:,1);
            f{this_node}.assy.y = xyz_new(:,2);
            f{this_node}.assy.z = xyz_new(:,3);
            clear xyz_old xyz_new; 
            
            % Transform keypoints
            xyz_old = [kp{this_node}.assy.x, kp{this_node}.assy.y, kp{this_node}.assy.z];
            xyz_old(:,4) = 1;
            xyz_new = (transf_est * xyz_old')';
            kp{this_node}.assy.x = xyz_new(:,1);
            kp{this_node}.assy.y = xyz_new(:,2);
            kp{this_node}.assy.z = xyz_new(:,3);
            clear xyz_old xyz_new; 
            
        else
            % Reversed order (sorted ascending)
            
            % Get transformation matrix
            transf_est = matching_pairwise{position}.transformation_A;
            
            % Transform fragment point cloud
            xyz_old = [f{this_node}.assy.x, f{this_node}.assy.y, f{this_node}.assy.z];
            xyz_old(:,4) = 1;
            xyz_new = (transf_est * xyz_old')';
            f{this_node}.assy.x = xyz_new(:,1);
            f{this_node}.assy.y = xyz_new(:,2);
            f{this_node}.assy.z = xyz_new(:,3);
            clear xyz_old xyz_new; 
            
            % Transform keypoints
            xyz_old = [kp{this_node}.assy.x, kp{this_node}.assy.y, kp{this_node}.assy.z];
            xyz_old(:,4) = 1;
            xyz_new = (transf_est * xyz_old')';
            kp{this_node}.assy.x = xyz_new(:,1);
            kp{this_node}.assy.y = xyz_new(:,2);
            kp{this_node}.assy.z = xyz_new(:,3);
            clear xyz_old xyz_new; 
        end
        
        transformation_temp{k} = transf_est;
    end % end loop k
    
    % Add up total transformation
    transformation_assy = eye(size(transformation_temp{1}));
    for k=1:length(transformation_temp)
    %for k=length(transformation_temp):-1:1
        transformation_assy = transformation_assy * transformation_temp{k};
    end
    % Save total transformation to fragment, keypoint structs
    f{this_node}.assy.transformation_assy = transformation_assy;
    kp{this_node}.assy.transformation_assy = transformation_assy;
end % end loop i


%% Plot reassembled fragments
if plot_result_flag == 1
    
    % Define colors
    for k=0:10
        my_colors{1+k*10} = 'r';
        my_colors{2+k*10} = 'b';
        my_colors{3+k*10} = 'g';
        my_colors{4+k*10} = 'c';
        my_colors{5+k*10} = 'm';
        my_colors{6+k*10} = 'k';
        my_colors{7+k*10} = 'r';
        my_colors{8+k*10} = 'b';
        my_colors{9+k*10} = 'g';
        my_colors{10+k*10} = 'c';
    end

    % Plot assembled fragments/keypoints
    % change from f to kp to plot keypoints
    figure;
    set(gcf,'color','w');
    title('Assembled fragments');
    for i=1:length(f)
        plot_3d(f,i,my_colors{i},'assy'); 
    end

    % Plot ground truth fragments/keypoints
    % change from f to kp to plot keypoints
    figure;
    set(gcf,'color','w');
    title('Ground truth fragments');
    for i=1:length(f)
        plot_3d(f,i,my_colors{i},'gt');
    end
    
    % Plot random pose fragments/keypoints
    figure;
    set(gcf,'color','w');
    title('Random pose fragments');
    for i=1:length(f)
        plot_3d(f,i,my_colors{i},'rp');
    end
end

%% Prepare output
f_out = f;
kp_out = kp;

%% save workspace
save workspace_assy_3d.mat;
save('cube_6_f_kp.mat','f','kp');

% End main
end

