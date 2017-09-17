function [dx, dwx, dwh, db] = rnn_backward_gpu(dout, cache)
%RNN_BACKWARD_GPU Compute the gradients of a recurrent layer on GPU
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
    dx = gpuArray.zeros(size(x)); % GPU
    dwx = gpuArray.zeros(size(wx)); dwh = gpuArray.zeros(size(wh)); db = gpuArray.zeros(size(b)); % GPU
    dhNext = gpuArray.zeros(size(out(:, :, 1))); % GPU
    for t = size(x, 3): -1 :1
        dh = dout(:, :, t) + dhNext;
        dhraw = (1 - out(:, :, t).^2) .* dh;  % Tanh Derivative
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