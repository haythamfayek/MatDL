function [out, cache] = affine_relu_forward(x, w, b)
%AFFINE_RELU_FORWARD Combo forward layer for fully-connected & ReLU layers
%   Inputs:
%       - x: input to layer, of size: batch size (m) x previous layer size
%       - w: weights, of size: previous layer size x layer size (l)
%       - b: biases, of size: 1 x layer size (l)
%   Outputs:
%       - out: output, of size: batch size (m) x layer size (l)
%       - cache: a structure of:
%           fcCache: cache for fully-connected layer
%           reluCache: cache for ReLU layer
%
% This file is part of the MaTDL toolbox and is made available under the 
% terms of the MIT license (see the LICENSE file) 
% from http://github.com/haythamfayek/MatDL
% Copyright (C) 2016-17 Haytham Fayek.

    [a, fcCache] = affine_forward(x, w, b);
    [out, reluCache] = relu_forward(a);
    cache.fcCache = fcCache; cache.reluCache = reluCache;
end
