function [A, q, p, Q] = getQqp_simTransf(C, G, S)

if ~exist('S','var'), S=eye(3); end

c1 = C(1); c2 = C(2); c3 = C(3);
g1 = G(1); g2 = G(2); g3 = G(3);

A = [c1 c2 c3 0 0 0 0 0 0 1 0 0 -g1;
     0 0 0 c1 c2 c3 0 0 0 0 1 0 -g2;
     0 0 0 0 0 0 c1 c2 c3 0 0 1 -g3];
 
% A = [  C'   0 0 0 0 0 0 1 0 0 -g1;
%      0 0 0   C'   0 0 0 0 1 0 -g2;
%      0 0 0 0 0 0   C'   0 0 1 -g3];

if nargout > 1
    % A' * A = [Q q/2; q'/2 p]
    At_A = A'*A;
    q = At_A(13,1:12)*2;
end

if nargout > 2
    p = At_A(13,13);
end

if nargout > 3
    Q = At_A(1:12, 1:12);
%     Q = Q + 10^(-13)*eye(12);
    Q = Q + 10^(-15)*eye(12);
end

end
