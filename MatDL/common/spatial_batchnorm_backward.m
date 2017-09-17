function [dx, dGamma, dBeta] = spatial_batchnorm_backward(dout, cache)
%SPATIAL_BATCHNORM_BACKWARD Wraps a batchnorm layer to compute the backward batchnorm for a convolutional layer
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

    [N, C, H, W] = size(dout);
    z = reshape(permute(dout, [1, 3, 4, 2]), (N * H * W), C);
    [z_out, dGamma, dBeta] = batchnorm_backward(z, cache);
    dx = permute(reshape(z_out, N, H, W, C), [1, 4, 2, 3]); 
end