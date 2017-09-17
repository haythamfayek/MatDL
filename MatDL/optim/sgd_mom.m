function [model, opt] = sgd_mom(model, grads, opt)
%SGD_MOM Do an SGD with momentum update step
%   Inputs:
%       - model: a structure of model parameters to update
%       - grads: a structure of gradients of model parameters
%       - opt: a structure of hyper-parameters and cache for sgd_mom
%   Outputs:
%       - model: a structure of updated model parameters
%       - opt: a structure of updated hyper-parameters and cache for sgd_mom
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
        dx = momentum * opt.vgrads.(p{i}) - opt.learningRate * grads.(p{i});
        opt.vgrads.(p{i}) = dx;
        model.(p{i}) = model.(p{i}) + dx;
    end

end

