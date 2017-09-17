function [dx] =  tanh_backward(dout, cache)
%TANH_BACKWARD Compute gradients for a tanh layer
%   Inputs:
%       - dout: upstream derivatives, of size: batch size (m) x layer size (l)
%       - cache: a structure of:
%           output: layer output
%   Outputs:
%       - dx: gradients w.r.t. x, of size: size(x)
%
% This file is part of the MaTDL toolbox and is made available under the
% terms of the MIT license (see the LICENSE file)
% from http://github.com/haythamfayek/MatDL
% Copyright (C) 2016-17 Haytham Fayek.

    out = cache.out; % x = cache.x;
    dx = dout .* (1 - (out.^2));
end
