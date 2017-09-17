function [dx, dw, db] = conv_relu_backward(dout, cache)
%CONV_RELU_BACKWARD Combo backward layer for convolutional & ReLU layers
%   Inputs:
%       - dout: upstream derivatives, of size: batch size (m) x ...
%               number of feature maps x feature map height x feature map width
%       - cache: a structure of:
%           convCache: cache for convolutional layer
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

    convCache = cache.convCache; reluCache = cache.reluCache;
    da = relu_backward(dout, reluCache);
    [dx, dw, db] = conv_backward(da, convCache);
end
