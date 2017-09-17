function [out, param, cache] = affine_bn_relu_forward(x, w, b, gamma, beta, param)
%AFFINE_BN_RELU_FORWARD Combo forward layer for fully-connected, batchnorm & ReLU layers
%   Inputs:
%       - x: input to layer, of size: batch size (m) x layer size (l)
%       - w: weights, of size: previous layer size x layer size (l)
%       - b: biases, of size: 1 x layer size (l)
%       - gamma: scale factor of size: 1 x layer size (l)
%       - beta: shift factor of size: 1 x layer size (l)
%       - param: a structure of:
%           mode: 'train' or 'test'
%           runningMean: running mean of size: 1 x layer size (l)
%           runningVar: running variance of size: 1 x layer size (l)
%   Outputs:
%       - out: output, of size: batch size (m) x layer size (l)
%       - param: updated param
%       - cache: a structure of:
%           fcCache: cache for fully-connected layer
%           bnCahce: cache for batchnorm layer
%           reluCache: cache for ReLU layer
%
% This file is part of the MaTDL toolbox and is made available under the 
% terms of the MIT license (see the LICENSE file) 
% from http://github.com/haythamfayek/MatDL
% Copyright (C) 2016-17 Haytham Fayek.

    [a, fcCache] = affine_forward(x, w, b);
    [abn, param, bnCache] = batchnorm_forward(a, gamma, beta, param);
    [out, reluCache] = relu_forward(abn);
    cache.fcCache = fcCache; cache.reluCache = reluCache; cache.bnCache = bnCache;
end
