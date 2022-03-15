function [T_32_gt,T_32_est,constraint] = triplewise_matching_3d(comb_pairwise,matching_pairwise,triple_index,th_R,th_T)
% Inputs:
% comb_pairwise (tuples, e.g. size 231x2), matching_pairwise (e.g. size 231x1),
% triple_index (triples, e.g. 1540x3), params th_R, th_T (FH: unclear)

% Config
plot_on = 1;

if plot_on == 1
    disp('----------- triplewise_matching_3d -----------');
    disp(['triple_index: ', num2str(triple_index)]);
end

index_12 = triple_index(:,1:2);
index_13 = [triple_index(:,1) triple_index(:,3)];
index_23 = triple_index(:,2:3);

% Find location of comb_pairwise in index_**
[~,loc_12] = ismember(index_12,comb_pairwise,'rows');
[~,loc_13] = ismember(index_13,comb_pairwise,'rows');
[~,loc_23] = ismember(index_23,comb_pairwise,'rows');

% T_32 = T12 * T31
T_32 = matching_pairwise{loc_23}.transformation_B;
T_12 = matching_pairwise{loc_12}.transformation_A;
T_31 = matching_pairwise{loc_13}.transformation_B;

% Check if any transformation matrix T contains NaN values
if (sum(sum(isnan(T_32))) > 0 || ...
    sum(sum(isnan(T_12))) > 0 || ...
    sum(sum(isnan(T_31))) > 0)

    % Prepare output
    constraint = 1;
    T_32_gt = [];
    T_32_est = [];

    if plot_on == 1        
        % Display
        disp('NaN values in T detected!');
        disp('T_32');
        disp(T_32);
        disp('T_12');
        disp(T_12);
        disp('T_31');
        disp(T_31);
    end
else

    T_32_gt = T_32;
    T_32_est = T_12*T_31;

    % Calculate angle between rotation matrices
    R_32_gt = T_32_gt(1:3, 1:3);
    R_32_est = T_32_est(1:3, 1:3);
    angle_R_32_gt = acos((trace(R_32_gt)-1)/2);
    angle_R_32_est = acos((trace(R_32_est)-1)/2);
    angle_diff = abs(angle_R_32_gt - angle_R_32_est);

    % Calculate distance between transformation matrices
    Transl_32_gt = T_32_gt(1:3,4);
    Transl_32_est = T_32_est(1:3,4);
    distance_diff = norm(Transl_32_gt - Transl_32_est);

    % Check if angle and distance difference are below threshold values
    if (angle_diff < th_R && distance_diff < th_T)
        constraint = 0;
        disp('Triplewise match!');
        disp(['Fragments: ', num2str(triple_index)]);
        disp(['angle difference (rad) = ',num2str(angle_diff)]);
        disp(['distance distance = ',num2str(distance_diff)]);
        disp('T_32:');
        disp(T_32);
        disp('T_12:');
        disp(T_12);
        disp('T_31:');
        disp(T_31);
    else
        disp('Angle and/or distance obove threshold!');
        constraint = 1;
    end

    if plot_on == 1
       disp('-----------------');
       disp('R_32_gt'); 
       disp(R_32_gt)

       disp('R_32_est');
       disp(R_32_est)

       disp('angle_R_32_gt');
       disp(angle_R_32_gt)

       disp('angle_R_32_est');
       disp(angle_R_32_est)

       disp('angle_diff');
       disp(angle_diff)

       disp('Transl_32_gt');
       disp(Transl_32_gt)

       disp('Transl_32_est');
       disp(Transl_32_est)

       disp('distance_diff');
       disp(distance_diff)

       disp('constraint');
       disp(constraint)

       if constraint == 0
           disp('Match!');
       else
           disp('No match.');
       end
    end

end % end if

end % end function