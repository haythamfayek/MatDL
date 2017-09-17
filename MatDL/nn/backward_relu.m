function [dx, dw, db] = backward_relu(dout, cache)
%BACKWARD_RELU Combo backward layer for fully-connected & ReLU layers
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

    reluCache = cache.reluCache; fcCache = cache.fcCache;
    da = relu_backward(dout, reluCache);
    [dx, dw, db] = backward(da, fcCache);
end