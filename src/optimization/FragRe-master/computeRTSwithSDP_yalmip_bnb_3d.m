function [R, t, s_opt, inliers, xopt, sol_yalmip] = ...
                computeRTSwithSDP_yalmip_bnb_3d(U, V, th, sLU, S, enfRot)

% Threshold (default values)
%if ~exist('th','var'), th = 10^(-5); end
if ~exist('th','var'), th = 0.5e-1; end

% bounds on scale parameters
%f ~exist('sLU','var'), sLU = [0.2 5.0]; end
if ~exist('sLU','var'), sLU = [0.9 1.1]; end


% covariance matrix
if ~exist('S','var'), S=eye(3); end

% enforce rotation matrix constraint
if ~exist('enfRot','var'), enfRot=1; end


%% Variables
dim = 12;
N = size(U,2); % number of putative assignment

x     = sdpvar(dim,1);
z     = binvar(N,1);

%s     = 1;
s = sdpvar(1);
BigM  = 10^(5);

% L constraint
if enfRot
    %L = eye(4) + conv_SO3(x);
    L = s*eye(4) + conv_SO3(x);
    Constraints = [ L >= 0 ];  % eq. (13)
else
    Constraints = [];
end


% L\infty constraints
for i = 1:N
    % residual:  r(x) == M*[x;1] % dim: 3-by-1
    M = getQqp_simTransf(U(:,i), V(:,i), S); % dim: 3-by-13
    Constraints = [Constraints, ...
                   norm(M*[x;1]) <= th + z(i)*BigM]; % Yalmip version of eq. (7)
end

% s constraints (bounds)
 lowerS  = sLU(1);
 uppderS = sLU(2);
 Constraints = [ Constraints, s>=lowerS, s<=uppderS ];


% Objective
Objective = sum(z);


%% Yalmip BnB options
options = sdpsettings('solver','bnb', 'verbose', 1);
%options.bnb.maxiter = 100000000;
options.bnb.maxiter = 100000;
options.bnb.method = 'best'; % 'best', 'breadth', 'depth', 'depthX'
options.bnb.gaptol = 1e-5; % Exit when (upper bound-lower bound)/(1e-3+abs(lower bound)) < gaptol
options.bnb.inttol = 1e-5; % tolerance for declaring a variable as integer
options.bnb.prunetol = 1e-05; % default: 1.0000e-04
% options.bnb.solver = 'sdpt3'; 
% options.bnb.solver = 'sedumi';
options.bnb.solver = 'mosek';    

    
%% Mosek solver options
acc = 1e-09;
%options.mosek.MSK_DPAR_INTPNT_CO_TOL_PFEAS = acc;
%options.mosek.MSK_DPAR_INTPNT_CO_TOL_DFEAS = acc;
%options.mosek.MSK_DPAR_INTPNT_CO_TOL_REL_GAP = acc;
% options.mosek.MSK_DPAR_ANA_SOL_INFEAS_TOL = 1e-10;


%% Optimize
sol = optimize(Constraints, Objective, options);


%% Result -- analyze error flags
if sol.problem == 0
    xopt = value(x); % extract 'x' value
else
     display('Something went wrong!');
     sol.info
     yalmiperror(sol.problem)
     R = NaN; s_opt = NaN; t = NaN; inliers = NaN; xopt = NaN; sol_yalmip = NaN;
     return;
end

% reshape solution
R = reshape(xopt(1:9),3,3)';
t = xopt(10:12);

% get scale and rot
%s_opt = s; % s_opt = abs(det(sR))^(1/3);   % s^3 = det(sR)
s_opt = abs(det(R))^(1/3);

% s_opt
% R = sR/s_opt;
    
% closest rotation
%     [U,D,V] = svd(sR);
%     R = U*[1 0 0; 0 1 0; 0 0 det(U*(V'))]*(V');
%     s_opt = det(D)^(1/3);

% return solution
inliers = ~ value(z);  % I = {i | z_i = 0}
sol_yalmip = sol;
end
