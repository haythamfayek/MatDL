function [dx, dwx, dwh, db] = rnn_backward(dout, cache)
%RNN_BACKWARD Compute the gradients of a recurrent layer
%c.f.: Karpathy's efficient python implementation: https://gist.github.com/karpathy/587454dc0146a6ae21fc
%   Inputs:
%       - dout: upstream derivatives, of size: batch size (m) x layer size (l) x time steps (t)
%       - cache: a structure of:
%           x: input to layer
%           hprev: initial hidden state
%           wx: input-to-hidden weights
%           wh: hidden-to-hidden weights
%           b: biases
%   Outputs:
%       - dx: gradients w.r.t. x, of size: size(x)
%       - dwx: gradients w.r.t. wx, of size: size(wx)
%       - dwh: gradients w.r.t. wh, of size: size(wh)
%       - db: gradients w.r.t. b, of size: size(b)
%
% This file is part of the MaTDL toolbox and is made available under the 
% terms of the MIT license (see the LICENSE file) 
% from http://github.com/haythamfayek/MatDL
% Copyright (C) 2016-17 Haytham Fayek.

    x = cache.x; hprev = cache.hprev; wx = cache.wx; wh = cache.wh; b = cache.b; out = cache.out;
    dx = zeros(size(x));
    dwx = zeros(size(wx)); dwh = zeros(size(wh)); db = zeros(size(b));
    dhNext = zeros(size(out(:, :, 1)));
    for t = size(x, 3): -1 :1
        dh = dout(:, :, t) + dhNext;
        dhraw = (1 - out(:, :, t).^2) .* dh;  % tanh Derivative
        dx(:, :, t) = dhraw * wx';
        dwx = dwx + x(:, :, t)' * dhraw;
        if (t ~= 1)
            dwh = dwh + out(:, :, t - 1)' * dhraw;
        elseif (t == 1)
            dwh = dwh + hprev' * dhraw;
        end
        db = db + sum(dhraw, 1);
        dhNext = dhraw * wh';
    end
end