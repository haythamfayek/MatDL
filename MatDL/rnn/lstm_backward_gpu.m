function [dX, dWLSTM, dc0, dh0] = lstm_backward_gpu(dHout_in, cache, dcn, dhn)
%LSTM_BACKWARD_GPU Compute the gradients of an LSTM layer on a GPU
%c.f.: Karpathy's efficient python implementation: https://gist.github.com/karpathy/587454dc0146a6ae21fc
%   Inputs:
%       - dout: upstream derivatives, of size: batch size (m) x layer size (l) x time steps (t)
%       - cache: a structure of variables for the backward pass
%   Outputs:
%       - dX: gradients w.r.t. X, of size: size(X)
%       - dWLSTM: gradients w.r.t. WLSTM, of size: size(WLSTM)
%       - dc0: gradients w.r.t. c0, of size: size(c0)
%       - dh0: gradients w.r.t. h0, of size: size(h0)
%
% This file is part of the MaTDL toolbox and is made available under the 
% terms of the MIT license (see the LICENSE file) 
% from http://github.com/haythamfayek/MatDL
% Copyright (C) 2016-17 Haytham Fayek.
    
    WLSTM = cache.WLSTM; out = cache.out;
    IFOGf = cache.IFOGf; IFOG = cache.IFOG;
    C = cache.C; Ct = cache.Ct; Hin = cache.Hin;
    c0 = cache.c0; % h0 = cache.h0;
    
    [M, H, T] = size(out);
    D = size(WLSTM, 1) - H - 1; % -1 for bias
    
    % backprop the LSTM
    dIFOG = gpuArray.zeros(size(IFOG));
    dIFOGf = gpuArray.zeros(size(IFOGf));
    dWLSTM = gpuArray.zeros(size(WLSTM));
    dHin = gpuArray.zeros(size(Hin));
    dC = gpuArray.zeros(size(C));
    dX = gpuArray.zeros(M, D, T);
    dh0 = gpuArray.zeros(M, H);
    dc0 = gpuArray.zeros(M, H);
    dHout = dHout_in;
    dC(:, :, T) = dC(:, :, T) + dcn;
    dHout(:, :, T) = dHout(:, :, T) + dhn;
    
    for t = T: -1 : 1
        tanhCt = Ct(:, :, t);
        dIFOGf(:, (2 * H + 1) : (3 * H), t) = tanhCt .* dHout(:, :, t);
        % backprop tanh non-linearity first then continue backprop
        dC(:, :, t) = dC(:, :, t) + (1 - tanhCt.^2) .* (IFOGf(:, (2 * H + 1) : (3 * H), t) .* dHout(:, :, t));
        
        if t > 1
            dIFOGf(:, (H + 1) : (2 * H), t) = C(:, :, t - 1) .* dC(:, :, t);
            dC(:, :, t - 1) = dC(:, :, t - 1) +  IFOGf(:, (H + 1) : (2 * H), t) .* dC(:, :, t);
        else
            dIFOGf(:, (H + 1) : (2 * H), t) = c0 .* dC(:, :, t);
            dc0 = IFOGf(:, (H + 1) : (2 * H), t) .* dC(:, :, t);
        end
        
        dIFOGf(:, 1 : H, t) = IFOGf(:, (3 * H + 1): end, t) .* dC(:, :, t);
        dIFOGf(:, (3 * H + 1) : end, t) = IFOGf(:, 1 : H, t) .* dC(:, :, t);
        
        % backprop activation functions
        dIFOG(:, (3 * H + 1) : end, t) = (1 - IFOGf(:, (3 * H + 1) : end, t).^ 2) .* dIFOGf(:, (3 * H + 1) : end, t);
        y = IFOGf(:, 1 : 3 * H, t);
        dIFOG(:, 1 : 3 * H, t) = (y .* (1.0 - y)) .* dIFOGf(:, 1 : 3 * H, t);

        % backprop matrix multiply
        dWLSTM = dWLSTM + Hin(:, :, t)' * dIFOG(:, :, t);
        dHin(:, :, t) = dIFOG(:, :, t) * WLSTM';

        % backprop the identity transforms into Hin
        dX(:, :, t) = dHin(:, 2 : D + 1, t);
        if (t > 1)
            dHout(:, :, t - 1) = dHout(:, :, t - 1) + dHin(:, (D + 1 + 1) : end, t);
        else
            dh0 = dh0 + dHin(:, (D + 1 + 1) : end, t);
        end
        
    end
    
end