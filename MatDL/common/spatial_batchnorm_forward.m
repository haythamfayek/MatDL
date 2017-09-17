function [out, param, cache] = spatial_batchnorm_forward(x, gamma, beta, param)
%SPATIAL_BATCHNORM_FORWARD Wraps batchnorm layer to compute the forward batchnorm for a convolutional layer
%   Inputs:
%       - x: input to layer of size batch size: (m) x channels (n) x height (H) x width (w)
%       - gamma: scale factor of size: 1 x (channels (n) * height (H) * width (w))
%       - beta: shift factor of size: 1 x (channels (n) * height (H) * width (w))
%       - param: a structure of:
%           mode: 'train' or 'test'
%           runningMean: running mean of size: 1 x layer size (l)
%           runningVar: running variance of size: 1 x layer size (l)
%   Outputs:
%       - out: normalized output, of size: size(x)
%       - param: updataed param
%       - cache: used for the backward pass
%
% This file is part of the MaTDL toolbox and is made available under the 
% terms of the MIT license (see the LICENSE file) 
% from http://github.com/haythamfayek/MatDL
% Copyright (C) 2016-17 Haytham Fayek.

    [N, C, H, W] = size(x);
    z = reshape(permute(x, [1, 3, 4, 2]), (N * H * W), C);
    [z_out, param, cache] = batchnorm_forward(z, gamma, beta, param);
    out = permute(reshape(z_out, N, H, W, C), [1, 4, 2, 3]);
end