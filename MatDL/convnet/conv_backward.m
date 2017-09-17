function [dx, dw, db] = conv_backward(dout, cache)
%BACKWARD Compute the gradients of a convolutional layer
%   Inputs:
%       - dout: upstream derivatives, of size: batch size (m) x ...
%               number of feature maps x feature map height x feature map width
%       - cache: a structure of:
%           x: input to layer
%           w: weights
%           b: biases
%           convParam: convolutional layer hyper-parameters (stride, padding, ..)
%   Outputs:
%       - dx: gradients w.r.t. x, of size: size(x)
%       - dw: gradients w.r.t. w, of size: size(w)
%       - db: gradients w.r.t. b, of size: size(b)
%
% This file is part of the MaTDL toolbox and is made available under the 
% terms of the MIT license (see the LICENSE file) 
% from http://github.com/haythamfayek/MatDL
% Copyright (C) 2016-17 Haytham Fayek.

    x = cache.x; w = cache.w; % b = cache.b; 
    convParam = cache.convParam; xCols = cache.xCols;
    stride = convParam.stride; pad = convParam.pad; useGPU = convParam.useGPU;
        
    db = sum(sum(sum(dout, 1), 3), 4);
    
    [numFilters, ~, filterHeight, filterWidth] = size(w);
    doutReshaped = reshape(permute(dout, [2, 1, 4, 3]), numFilters, []);
    dw = reshape((doutReshaped * xCols')', [size(w,4), size(w,3), size(w,2), size(w,1)]);
    dw = permute(dw, [4, 3, 2, 1]);

    dxCols = reshape(permute(w, [1 4 3 2]), numFilters, [])' * doutReshaped;
    
    if(useGPU)
        dx = gpuArray(col2im_c(gather(dxCols), size(x,1), size(x,2), size(x,3), size(x,4), filterHeight, filterWidth, pad, stride));
    else
        dx = col2im_c(dxCols, size(x,1), size(x,2), size(x,3), size(x,4), filterHeight, filterWidth, pad, stride);
    end

end