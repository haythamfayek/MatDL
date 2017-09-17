function [dx, dGamma, dBeta] = batchnorm_backward(dout, cache)
%BATCHNORM_BACKWARD Compute gradients for a batchnorm layer
%   Inputs:
%       - dout: upstream derivatives
%       - cache: a structure of:
%           param: layer parameters
%           x: input to layer
%           xn: normalized input
%           gamma: scaling factor
%           batchStd: batch standard deviation
%   Outputs:
%       - dx: gradients w.r.t. x, of size: size(x)
%       - dGamma: gradients w.r.t. gamma, of size: size(gamma)
%       - dBeta: gradients w.r.t. beta, of size: size(beta)
%
% This file is part of the MaTDL toolbox and is made available under the 
% terms of the MIT license (see the LICENSE file) 
% from http://github.com/haythamfayek/MatDL
% Copyright (C) 2016-17 Haytham Fayek.

    param = cache.param;
    mode = param.mode;
    
    x = cache.x; xn = cache.xn;
    gamma = cache.gamma; % beta = cache.beta;
    batchStd = cache.batchStd;
    m = size(x, 1);
    
    if strcmp(mode, 'train')
        
        xc = cache.xc;
    
        dxn = repmat(gamma, m, 1) .* dout;
        dxc = dxn ./ repmat(batchStd, m, 1);
        dStd = -sum((dxn .* xc) ./ repmat(batchStd.^2, m, 1), 1);
        dVar = 0.5 .* dStd ./ batchStd;
        dxc = dxc + ((2 / m) .* xc .* repmat(dVar, m, 1));
        dMean = sum(dxc, 1);
        dx = dxc - (repmat(dMean, m, 1) ./ m);

        dGamma = sum(xn .* dout, 1);
        dBeta = sum(dout, 1);
        
    elseif strcmp(mode, 'test')
        
        dxn = repmat(gamma, m, 1) .* dout;
        dx = dxn ./ repmat(batchStd, m, 1);
        
        dGamma = sum(xn .* dout, 1);
        dBeta = sum(dout, 1);  
        
    else
        error('BatchNorm Malfunction');
    end
    
end