function [out, param, cache] = batchnorm_forward(x, gamma, beta, param)
%BATCHNORM_FORWARD Compute forward pass for a batchnorm layer
%   Inputs:
%       - x: input to layer, of size: batch size (m) x layer size (l)
%       - gamma: scale factor, of size: 1 x layer size (l)
%       - beta: shift factor, of size: 1 x layer size (l)
%       - param: a structure of:
%           mode: 'train' or 'test'
%           runningMean: running mean, of size: 1 x layer size (l)
%           runningVar: running variance, of size: 1 x layer size (l)
%   Outputs:
%       - out: normalized output, of size: size(x)
%       - param: updataed param
%       - cache: a structure of cached variables for the backward pass
%
% This file is part of the MaTDL toolbox and is made available under the 
% terms of the MIT license (see the LICENSE file) 
% from http://github.com/haythamfayek/MatDL
% Copyright (C) 2016-17 Haytham Fayek.

    mode = param.mode;
    runningMean = param.runningMean; runningVar = param.runningVar;
    momentum = 0.9;
    m = size(x, 1);
    
    if strcmp(mode, 'train')
        batchMean = mean(x, 1);
        xc = (x - repmat(batchMean, m, 1));
        batchVar = mean(xc.^2, 1);
        batchStd = sqrt(batchVar + eps);
        xn = xc ./ repmat(batchStd, m, 1);
        out = repmat(gamma, m, 1) .* xn + repmat(beta, m, 1);
        
        runningMean = momentum * runningMean + (1 - momentum) * batchMean;
        runningVar = momentum * runningVar + (1 - momentum) * batchVar;
        
        param.runningMean = runningMean;
        param.runningVar = runningVar;
        
        cache.xc = xc;
        
    elseif strcmp(mode, 'test')
        batchStd = sqrt(runningVar + eps);
        xn =  (x - repmat(runningMean, m, 1)) ./ repmat(batchStd, m, 1);
        out = repmat(gamma, m, 1) .* xn + repmat(beta, m, 1);
        
    else
        error('BatchNorm Malfunction');
    end
    
    cache.x = x;
    cache.xn = xn;
    cache.gamma = gamma;
    cache.beta = beta;
    cache.param = param;
    cache.batchStd = batchStd;
end