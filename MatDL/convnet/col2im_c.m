function xPadded = col2im_c(cols, M, C, H, W, filterHeight, filterWidth, pad, stride)
%COL2IM_C Reshape columns into image blocks using a mex function
%   Inputs:
%       - cols: columns to be reshaped into image blocks
%       - M: number of images
%       - C: number of channels
%       - H: height of each image
%       - W: width of each image
%       - filterHeight: layer filter height
%       - filterWidth: layer filter width
%       - pad: layer padding
%       - stride: layer convolution stride
%   Outputs:
%       - xPadded: image blocks
%
% This file is part of the MaTDL toolbox and is made available under the 
% terms of the MIT license (see the LICENSE file) 
% from http://github.com/haythamfayek/MatDL
% Copyright (C) 2016-17 Haytham Fayek.

    HH = (H + 2 * pad - filterHeight) / stride + 1;
    WW = (W + 2 * pad - filterWidth) / stride + 1;
    
    xPadded = zeros(M, C, H + 2 * pad, W + 2 * pad);
    
    xPadded = col2im_mex(cols, M, C, H, W, HH, WW, filterHeight, filterWidth, pad, stride, xPadded);

    if pad > 0
        xPadded = xPadded(:, :, (pad + 1 : end - pad), (pad + 1 : end - pad)); 
    end
    
end