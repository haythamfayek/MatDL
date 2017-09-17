function [out] = softmax_forward(x)
%SOFTMAX_FORWARD Compute the forward pass for a softmax layer
%   Inputs:
%       - x: logits, of size: batch size (m) x number of classes (k)
%   Outputs:
%       - out: class probabilities, of size: batch size (m) x number of classes (k)
%
% This file is part of the MaTDL toolbox and is made available under the 
% terms of the MIT license (see the LICENSE file) 
% from http://github.com/haythamfayek/MatDL
% Copyright (C) 2016-17 Haytham Fayek.

    K = size(x, 2);
    out = exp(x - repmat(max(x, [], 2), 1, K));
    out = out ./ repmat(sum(out, 2), 1, K);
end