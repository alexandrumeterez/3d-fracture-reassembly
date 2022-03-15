% semidefinite rotation constraint
% A 3x3 matrix A lies in conv(SO(3)) if and only if eye(4)+conv_SO3(A) >= 0
function L = conv_SO3(x)
    x11 = x(1); x12 = x(2); x13 = x(3);
    x21 = x(4); x22 = x(5); x23 = x(6);
    x31 = x(7); x32 = x(8); x33 = x(9);
    
    L = [ x11+x22+x33,   x32-x23  ,   x13-x31   ,   x21-x12   ; ...
           x32-x23   , x11-x22-x33,   x21+x12   ,   x13+x31   ; ...
           x13-x31   ,   x21+x12  ,  x22-x11-x33,   x32+x23   ; ...
           x21-x12   ,   x13+x31  ,   x32+x23   , x33-x11-x22];
end
