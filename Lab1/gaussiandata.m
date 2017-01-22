function [patterns, targets, gridsize, x, y, z] = gaussiandata()
    x=[-5:1:5]';
    y=x;
    z=exp(-x.*x*0.1) * exp(-y.*y*0.1)' - 0.5;
    [z_row, z_col] = size(z);
    ndata = z_row*z_col;

    targets = reshape (z, 1, ndata);
    [xx, yy] = meshgrid (x, y);

    patterns = [reshape(xx, 1, ndata); reshape(yy, 1, ndata)];
    gridsize = z_row;
end
