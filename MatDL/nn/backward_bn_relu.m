function [dx, dw, db, dGamma, dBeta] = backward_bn_relu(dout, cache)
%BACKWARD_BN_RELU Combo backward layer for fully-connected, batchnorm & ReLU layers
%   Inputs:
%       - dout: upstream derivatives, of size: batch size (m) x layer size (l)
%       - cache: a structure of:
%           fcCache: cache for fully-connected layer
%           bnCahce: cache for batchnorm layer
%           reluCache: cache for ReLU layer
%   Outputs:
%       - dx: gradients w.r.t. x, of size: size(x)
%       - dw: gradients w.r.t. w, of size: size(w)
%       - db: gradients w.r.t. b, of size: size(b)
%       - dGamma: gradients w.r.t. gamma, of size: size(gamma)
%       - dBeta: gradients w.r.t. beta, of size: size(beta)
%
% This file is part of the MaTDL toolbox and is made available under the 
% terms of the MIT license (see the LICENSE file) 
% from http://github.com/haythamfayek/MatDL
% Copyright (C) 2016-17 Haytham Fayek.

    reluCache = cache.reluCache; fcCache = cache.fcCache; bnCache = cache.bnCache;
    da = relu_backward(dout, reluCache);
    [dabn, dGamma, dBeta] = batchnorm_backward(da, bnCache);
    [dx, dw, db] = backward(dabn, fcCache);
end