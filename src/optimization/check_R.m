function [n] = check_R(R)
% Check rotation matrix

% Test vector
v = [1;0;0];

n = norm(R*v);

disp('Rotation matrix R: ');
disp(R);

disp(['Test vector v: ',num2str(v')]);
disp(['Norm (length) v: ',num2str(norm(v))]);
disp(['Norm (length) R*v: ', num2str(norm(R*v))]); 
