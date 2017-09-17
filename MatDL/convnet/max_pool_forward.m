function [out, cache] = max_pool_forward(x, poolParam)
%MAX_POOL_FORWARD Compute the forward pass for a max pooling layer
%   Inputs:
%       - x: input to layer, of size: batch size (m) x ...
%               number of input channels x channel height x channel width
%       - poolParam: a structure of:
%           poolHeight: window height
%           poolWidth: window width
%           stride: convolution stride
%           useGPU: GPU flag
%   Outputs:
%       - out: output, of size: batch size (m) x number of filters x (depends on hyper-parameters)
%       - cache: a structure of:
%           x: input to layer
%           xCols: rehshaped input
%           xColsArgmax: rehshaped input after max
%           poolParam: (see above)
%
% This file is part of the MaTDL toolbox and is made available under the 
% terms of the MIT license (see the LICENSE file) 
% from http://github.com/haythamfayek/MatDL
% Copyright (C) 2016-17 Haytham Fayek.

    [M, C, H, W] = size(x);
    poolHeight = poolParam.poolHeight; poolWidth = poolParam.poolWidth;
    stride = poolParam.stride;
    useGPU = poolParam.useGPU;
    
    assert(mod(H - poolHeight, stride) == 0, 'Invalid height');
    assert(mod(W - poolWidth, stride) == 0, 'Invalid width');
    
    outHeight = (H - poolHeight) / stride + 1;
    outWidth = (W - poolWidth) / stride + 1;
    
    x = permute(x,[2, 1, 3, 4]); xSplit = reshape(x, M * C, 1, H, W);
    
    if(useGPU)
        xCols = im2col_c(gather(xSplit), poolHeight, poolWidth, 0, stride);
        [~, xColsArgmax] = max(xCols, [], 1); 
        xColsArgmax = gpuArray(xColsArgmax);
    else
        xCols = im2col_c(xSplit, poolHeight, poolWidth, 0, stride);
        [~, xColsArgmax] = max(xCols, [], 1); 
    end
        
    idx = sub2ind(size(xCols), xColsArgmax, 1:size(xCols,2));
    xColsMax = xCols(idx);
    % out = permute(reshape(x_cols_max, out_height, out_width, M, C), [3, 4, 1, 2]);
    out = permute(reshape(xColsMax', C, M, outWidth, outHeight), [2, 1, 4, 3]);
    
    cache.x = x; cache.xCols = xCols; 
    cache.xColsArgmax = xColsArgmax; cache.poolParam = poolParam;

end
