function [out, bnParam, cache] =  conv_bn_relu_forward(x, w, b, gamma, beta, bnParam, convParam)
%CONV_BN_RELU_FORWARD Combo forward layer for convolutional, batchnorm & ReLU layers
%   Inputs:
%       - x: input to layer, of size: batch size (m) x ...
%               number of input channels x channel height x channel width
%       - w: weights, of size: number of channels in previous layer x number of filters x ...
%                               filter height x filter width
%       - b: biases, of size: 1 x number of filters
%       - gamma: scale factor of size: 1 x number of filters
%       - beta: shift factor of size: 1 x number of filters
%       - bnParam: a structure of:
%           mode: 'train' or 'test'
%           runningMean: running mean of size: 1 x number of filters
%           runningVar: running variance of size: 1 x number of filters
%       - convParam: a structure of:
%           stride: convolution stride
%           pad: convolution padding
%           useGPU: GPU flag
%   Outputs:
%       - out: output, of size: batch size (m) x number of filters x (depends on hyper-parameters)
%       - bnParam: updated bnParam
%       - cache: a structure of:
%           convCache: cache for convolutional layer
%           bnCahce: cache for batchnorm layer
%           reluCache: cache for ReLU layer
%
% This file is part of the MaTDL toolbox and is made available under the 
% terms of the MIT license (see the LICENSE file) 
% from http://github.com/haythamfayek/MatDL
% Copyright (C) 2016-17 Haytham Fayek.

    [a, convCache] = conv_forward(x, w, b, convParam);
    [abn, bnParam, bnCache] = spatial_batchnorm_forward(a, gamma, beta, bnParam);
    [out, reluCache] = relu_forward(abn);
    cache.convCache = convCache; cache.reluCache = reluCache;
    cache.bnCache = bnCache;
end