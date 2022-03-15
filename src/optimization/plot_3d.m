function [fig_handle] = plot_3d(f,f_index,color,plot_type)
% Plot_type: rp (random pose), gt (ground truth) or assy (assembled)

if nargin < 3
    color = 'b';
end

try
    switch plot_type
        case 'rp'
            x = f{f_index}.rp.x;
            y = f{f_index}.rp.y;
            z = f{f_index}.rp.z;
        case 'gt'
            x = f{f_index}.gt.x;
            y = f{f_index}.gt.y;
            z = f{f_index}.gt.z;
        case 'assy'
            x = f{f_index}.assy.x;
            y = f{f_index}.assy.y;
            z = f{f_index}.assy.z;
        otherwise
            error(['Unknown plot type (',plot_type,')']);
    end

    hold on;
    grid on;
    axis equal;
    plot3(x,y,z,[color,'.']');
catch
    disp(['Skip plot of fragment ',num2str(f_index)]);
end
