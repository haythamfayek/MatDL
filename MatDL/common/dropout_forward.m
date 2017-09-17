function [out, cache] = dropout_forward(x, param)
%DROPOUT_FORWARD Compute the forward pass for a dropout layer
%   Inputs:
%       - x: input to layer, of size: batch size (m) x layer size (l)
%       - param: a structure of:
%           mode: 'train' or 'test'
%           p: keep probability [0,1]
%           useGPU: flag to use GPU
%   Outputs:
%       - out: output, of size: batch size (m) x layer size (l)
%       - cache: a structure of:
%           mask: mask used to drop units, of size: batch size (m) x layer size (l)
%           param: cache of param (see above)
%
% This file is part of the MaTDL toolbox and is made available under the 
% terms of the MIT license (see the LICENSE file) 
% from http://github.com/haythamfayek/MatDL
% Copyright (C) 2016-17 Haytham Fayek.

    mode = param.mode; p = param.p; useGPU = param.useGPU;

    if strcmp(mode, 'train')
        if (useGPU), mask = (gpuArray.rand(size(x)) < p) / p;
        else, mask = (rand(size(x)) < p) / p; end
        out = x .* mask;
        cache.mask = mask;
    
    elseif strcmp(mode, 'test')
        out = x;
        
    else
        error('DropOut Malfunction');
    end
    
    cache.param = param;
end