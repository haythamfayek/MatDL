function [out, cache] =  sigmoid_forward(x)
%SIGMOID_FORWARD Compute the forward pass for a sigmoid layer
%   Inputs:
%       - x: input to layer, of size: batch size (m) x layer size (l)
%   Outputs:
%       - out: output, of size: batch size (m) x layer size (l)
%       - cache: a structure of:
%           out: layer output
%
% This file is part of the MaTDL toolbox and is made available under the 
% terms of the MIT license (see the LICENSE file) 
% from http://github.com/haythamfayek/MatDL
% Copyright (C) 2016-17 Haytham Fayek.

    out = 1 ./ (1 + exp(-x));
    cache.out = out; % cache.x = x;
end