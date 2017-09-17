function [out, cache] = forward(x, w, b)
%FORWARD Compute the forward pass for a fully-connected layer
%   Inputs:
%       - x: input to layer, of size: batch size (m) x previous layer size
%       - w: weights, of size: previous layer size x layer size (l)
%       - b: biases, of size: 1 x layer size (l)
%   Outputs:
%       - out: output, of size: batch size (m) x layer size (l)
%       - cache: a structure of:
%           x: input to layer
%           w: weights
%           b: biases
%
% This file is part of the MaTDL toolbox and is made available under the 
% terms of the MIT license (see the LICENSE file) 
% from http://github.com/haythamfayek/MatDL
% Copyright (C) 2016-17 Haytham Fayek.

    out = x * w + repmat(b, size(x,1), 1);
    cache.x = x; cache.w = w; cache.b = b;
end

