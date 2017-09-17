function [out, cache] = conv_forward(x, w, b, convParam)
%CONV_FORWARD Compute the forward pass for a convolutional layer
%   Inputs:
%       - x: input to layer, of size: batch size (m) x ...
%               number of input channels x channel height x channel width
%       - w: weights, of size: number of channels in previous layer x number of filters x ...
%                               filter height x filter width
%       - b: biases, of size: 1 x number of filters
%       - convParam: a structure of:
%           stride: convolution stride
%           pad: convolution padding
%           useGPU: GPU flag
%   Outputs:
%       - out: output, of size: batch size (m) x number of filters x (depends on hyper-parameters)
%       - cache: a structure of:
%           x: input to layer
%           w: weights
%           b: biases
%           convParam: (see above)
%
% This file is part of the MaTDL toolbox and is made available under the 
% terms of the MIT license (see the LICENSE file) 
% from http://github.com/haythamfayek/MatDL
% Copyright (C) 2016-17 Haytham Fayek.

    [M, ~, H, W] = size(x);
    [numFilters, ~, filterHeight, filterWidth] = size(w);
    stride = convParam.stride; pad = convParam.pad; useGPU = convParam.useGPU;
    
    % Check Dimensions
    assert(mod(W + 2 * pad - filterWidth, stride) == 0, 'Width does not fit');
    assert(mod(H + 2 * pad - filterHeight, stride) == 0, 'Height does not fit');
    
    % Create Output
    outHeight = (H + 2 * pad - filterHeight) / stride + 1;
    outWidth = (W + 2 * pad - filterWidth) / stride + 1;
    
    if(useGPU)
        out = gpuArray.zeros(M, numFilters, outHeight, outWidth);
        xCols = gpuArray(im2col_c(gather(x), size(w, 3), size(w, 4), pad, stride));
    else
        out = zeros(M, numFilters, outHeight, outWidth);
        xCols = im2col_c(x, size(w, 3), size(w, 4), pad, stride);
    end    
    
    % res = bsxfun(@plus, reshape(w, size(w, 1), []) * x_cols, b');
    res = bsxfun(@plus, reshape(permute(w, [1 4 3 2]), size(w, 1), []) * xCols, b');
    
    out = reshape(res', size(x, 1), size(out, 4), size(out, 3), size(w, 1));
    out = permute(out, [1, 4, 3, 2]);

    cache.x = x; cache.w = w; cache.b =  b; 
    cache.convParam = convParam; cache.xCols = xCols;

end