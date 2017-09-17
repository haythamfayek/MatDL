function [dx] = relu_backward(dout, cache)
%RELU_BACKWARD Compute gradients for a Rectified Linear Unit (ReLU) layer
%   Inputs:
%       - dout: upstream derivatives, of size: batch size (m) x layer size (l)
%       - cache: a structure of:
%           x: input to layer
%   Outputs:
%       - dx: gradients w.r.t. x, of size: size(x)
%
% This file is part of the MaTDL toolbox and is made available under the
% terms of the MIT license (see the LICENSE file)
% from http://github.com/haythamfayek/MatDL
% Copyright (C) 2016-17 Haytham Fayek.

    x = cache.x;
    dx = dout .* (x > 0);
end
