function [model, opt] = nestrov(model, grads, opt)
%NESTROV Do a Nestrov-momentum SGD update step
%   Inputs:
%       - model: a structure of model parameters to update
%       - grads: a structure of gradients of model parameters
%       - opt: a structure of hyper-parameters and cache for nestrov
%   Outputs:
%       - model: a structure of updated model parameters
%       - opt: a structure of updated hyper-parameters and cache for nestrov
%
% This file is part of the MaTDL toolbox and is made available under the 
% terms of the MIT license (see the LICENSE file) 
% from http://github.com/haythamfayek/MatDL
% Copyright (C) 2016-17 Haytham Fayek.

    % Set Variable Momentum
    if (opt.epoch <= opt.switchEpochMomentum)
        momentum = opt.initialMomentum;
    else
        momentum = opt.finalMomentum; 
    end
    
    p = fieldnames(model);
    for i = 1:numel(p)
        v = opt.vgrads.(p{i});
        opt.vgrads.(p{i}) = momentum * opt.vgrads.(p{i}) - opt.learningRate * grads.(p{i});
        dx = - momentum * v + (1 + momentum) * opt.vgrads.(p{i});
        model.(p{i}) = model.(p{i}) + dx;
    end

end

