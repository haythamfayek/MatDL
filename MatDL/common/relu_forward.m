function [out, cache] = relu_forward(x)
%RELU_FORWARD Compute the forward pass for a Rectified Linear Unit (ReLU) layer
%   Inputs:
%       - x: input to layer, of size: batch size (m) x layer size (l)
%   Outputs:
%       - out: output, of size: batch size (m) x layer size (l)
%       - cache: a structure of:
%           x: input to layer
%
% This file is part of the MaTDL toolbox and is made available under the 
% terms of the MIT license (see the LICENSE file) 
% from http://github.com/haythamfayek/MatDL
% Copyright (C) 2016-17 Haytham Fayek.

    out = max(0, x);
    cache.x = x;
end