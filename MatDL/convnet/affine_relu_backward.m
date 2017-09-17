function [dx, dw, db] = affine_relu_backward(dout, cache)
%AFFINE_RELU_BACKWARD Combo backward layer for fully-connected & ReLU layers
%   Inputs:
%       - dout: upstream derivatives, of size: batch size (m) x layer size (l)
%       - cache: a structure of:
%           fcCache: cache for fully-connected layer
%           reluCache: cache for ReLU layer
%   Outputs:
%       - dx: gradients w.r.t. x, of size: size(x)
%       - dw: gradients w.r.t. w, of size: size(w)
%       - db: gradients w.r.t. b, of size: size(b)
%
% This file is part of the MaTDL toolbox and is made available under the 
% terms of the MIT license (see the LICENSE file) 
% from http://github.com/haythamfayek/MatDL
% Copyright (C) 2016-17 Haytham Fayek.

    fcCache = cache.fcCache; reluCache = cache.reluCache;
    da = relu_backward(dout, reluCache);
    [dx, dw, db] = affine_backward(da, fcCache);
end