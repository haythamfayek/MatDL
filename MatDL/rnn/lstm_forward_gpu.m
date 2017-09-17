function [out, cend, houtend, cache] = lstm_forward_gpu(x, h0, WLSTM, c0)
%LSTM_FORWARD_GPU Compute the forward pass for an LSTM layer on a GPU
%c.f.: Karpathy's efficient python implementation: https://gist.github.com/karpathy/587454dc0146a6ae21fc
%   Inputs:
%       - x: input to layer, of size: batch size (m) x previous layer size x time steps (t) 
%       - h0: hidden state from previous (initial) time step, of size: batch size (m) x layer size (l)
%       - WLSTM: LSTM weights, of size: previous layer size x (layer size(l) * 4)
%       - c0: cell activiations from previous (initial) time step, of size: batch size (m) x layer size (l)
%   Outputs:
%       - out: output, of size: batch size (m) x layer size (l) x time steps (t)
%       - cache: a structure of variables for the backward pass
%
% This file is part of the MaTDL toolbox and is made available under the 
% terms of the MIT license (see the LICENSE file) 
% from http://github.com/haythamfayek/MatDL
% Copyright (C) 2016-17 Haytham Fayek.

    [M, D, T] = size(x); % Batchsize, Input Dimension, Time Steps
    H = size(WLSTM, 2) / 4;
    xphpb = size(WLSTM, 1); % x plus h plus bias

    Hin = gpuArray.zeros(M, xphpb, T); % input [1, xt, ht-1] to each tick of the LSTM
    out = gpuArray.zeros(M, H, T); % hidden representation of the LSTM (gated cell content)
    IFOG = gpuArray.zeros(M, H * 4, T); % input, forget, output, gate (IFOG)
    IFOGf = gpuArray.zeros(M, H * 4, T); % after nonlinearity
    C = gpuArray.zeros(M, H, T); % cell content
    Ct = gpuArray.zeros(M, H, T); % tanh of cell content
    
    for t = 1:T
        if (t > 1), hprev = out(:, :, t - 1); else hprev = h0; end
        Hin(:, 1, t) = 1; % bias
        Hin(:, 2 : (D + 1), t) = x(:, :, t); % D + 1 to account for bias
        Hin(:, (D + 1 + 1): end, t) = hprev;
        % Compute all gate activations. Most work is this line
        IFOG(:, :, t) = Hin(:, :, t) * WLSTM;
        % Non-linearities
        IFOGf(:, 1 : (3 * H), t) = 1 ./ (1 + exp(-IFOG(:, 1 : (3 * H), t))); % sigmoids
        IFOGf(:, (3 * H + 1) : end, t) = tanh(IFOG(:, (3 * H + 1) : end, t)); % tanh
        % Compute the cell activation
        if (t > 1), cprev = C(:, :, t - 1); else cprev = c0; end
        C(:, :, t) = IFOGf(:, 1 : H, t) .* IFOGf(:, (3 * H + 1) : end, t) + IFOGf(:, (H + 1) : (2 * H), t) .* cprev;
        Ct(:, :, t) = tanh(C(:, :, t));
        out(:, :, t) = IFOGf(:, (2 * H + 1) : (3 * H), t) .* Ct(:, :, t);      
    end
    
    cache.WLSTM = WLSTM; cache.out = out; 
    cache.IFOGf = IFOGf; cache.IFOG = IFOG; 
    cache.C = C; cache.Ct = Ct; cache.Hin = Hin;
    cache.c0 = c0; cache.h0 = h0;
    
    cend = C(:, :, end); houtend = out(:, :, end);
    
end