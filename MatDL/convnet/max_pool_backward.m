function [dx] = max_pool_backward(dout, cache)
%MAX_POOL_BACKWARD Compute the gradients through a max pool layer
%   Inputs:
%       - dout: upstream derivatives, of size: batch size (m) x ...
%               number of feature maps x feature map height x feature map width
%       - cache: a structure of:
%           x: input to layer
%           xCols: rehshaped input
%           xColsArgmax: rehshaped input after max
%           poolParam: (see above)
%   Outputs:
%       - dx: gradients w.r.t. x, of size: size(x)
%
% This file is part of the MaTDL toolbox and is made available under the 
% terms of the MIT license (see the LICENSE file) 
% from http://github.com/haythamfayek/MatDL
% Copyright (C) 2016-17 Haytham Fayek.

    x = cache.x; xCols = cache.xCols;
    xColsArgmax = cache.xColsArgmax; poolParam = cache.poolParam;
    [M, C, H, W] = size(x);
    poolHeight = poolParam.poolHeight; poolWidth = poolParam.poolWidth;
    stride = poolParam.stride;
    useGPU = poolParam.useGPU;
    
    % dout_reshaped = permute(dout, [3, 4, 1, 2]);
    doutReshaped = permute(dout, [2, 1, 4, 3]);
    doutReshaped = doutReshaped(:);
    
    if(useGPU)
        dxCols = gpuArray.zeros(size(xCols));
        idx = sub2ind(size(dxCols), xColsArgmax, 1:size(dxCols,2));
        dxCols(idx) = doutReshaped;
        dx = gpuArray(col2im_c(gather(dxCols), M * C, 1, H, W, poolHeight, poolWidth, 0, stride));
    else
        dxCols = zeros(size(xCols));
        idx = sub2ind(size(dxCols), xColsArgmax, 1:size(dxCols,2));
        dxCols(idx) = doutReshaped;
        dx = col2im_c(dxCols, M * C, 1, H, W, poolHeight, poolWidth, 0, stride);
    end
    dx = reshape(dx, size(x));
    dx = permute(dx, [2, 1, 3, 4]);
    
end
