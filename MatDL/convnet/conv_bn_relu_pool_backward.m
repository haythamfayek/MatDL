function [dx, dw, db, dGamma, dBeta] = conv_bn_relu_pool_backward(dout, cache)
%CONV_BN_RELU_POOL_BACKWARD Combo backward layer for convolutional, batchnorm, ReLU & pool layers
%   Inputs:
%       - dout: upstream derivatives, of size: batch size (m) x ...
%               number of feature maps x feature map height x feature map width
%       - cache: a structure of:
%           convCache: cache for convolutional layer
%           bnCahce: cache for batchnorm layer
%           reluCache: cache for ReLU layer
%           poolCache: cache for pool layer
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

    convCache = cache.convCache;
    reluCache = cache.reluCache;
    poolCache = cache.poolCache;
    bnCache = cache.bnCache;
    ds = max_pool_backward(dout, poolCache);
    da = relu_backward(ds, reluCache);
    [dabn, dGamma, dBeta] = spatial_batchnorm_backward(da, bnCache);
    [dx, dw, db] = conv_backward(dabn, convCache);
end
