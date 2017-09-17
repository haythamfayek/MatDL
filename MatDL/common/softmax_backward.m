function [loss, dx] = softmax_backward(x, y)
%SOFTMAX_BACKWARD Compute the cross entropy loss & gradients for a softmax layer
%   Inputs:
%       - x: class probabilities, of size: batch size (m) x number of classes (k)
%       - y: labels, of size: batch size (m) x number of classes (k)
%   Outputs:
%       - loss: cross entropy loss
%       - dx: gradients w.r.t. x, of size: size(x)
%
% This file is part of the MaTDL toolbox and is made available under the
% terms of the MIT license (see the LICENSE file)
% from http://github.com/haythamfayek/MatDL
% Copyright (C) 2016-17 Haytham Fayek.

    m = size(x, 1);
    loss = -sum(sum(y .* log(x + eps))) / m;
    dx = (x - y) / m;
end
