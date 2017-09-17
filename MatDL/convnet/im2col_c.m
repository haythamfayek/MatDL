function cols = im2col_c(x, filterHeight, filterWidth, pad, stride)
%IM2COL_C Reshape image blocks into columns using a mex function
%   Inputs:
%       - x: image blocks
%       - filterHeight: layer filter height
%       - filterWidth: layer filter width
%       - pad: layer padding
%       - stride: layer convolution stride
%   Outputs:
%       - cols: columns
%
% This file is part of the MaTDL toolbox and is made available under the 
% terms of the MIT license (see the LICENSE file) 
% from http://github.com/haythamfayek/MatDL
% Copyright (C) 2016-17 Haytham Fayek.

    [M, C, H, W] = size(x);
    HH = (H + 2 * pad - filterHeight) / stride + 1;
    WW = (W + 2 * pad - filterWidth) / stride + 1;
    
    xPadded = padarray(x, [0, 0, pad, pad]);
    
    cols = im2col_mex(xPadded, M, C, H, W, HH, WW, filterHeight, filterWidth, pad, stride);
    
end