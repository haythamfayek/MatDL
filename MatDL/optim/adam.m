function [model, opt] = adam(model, grads, opt)    
%ADAM Do an ADAM update step
%   Inputs:
%       - model: a structure of model parameters to update
%       - grads: a structure of gradients of model parameters
%       - opt: a structure of hyper-parameters and cache for ADAM
%   Outputs:
%       - model: a structure of updated model parameters
%       - opt: a structure of updated hyper-parameters and cache for ADAM
%
% This file is part of the MaTDL toolbox and is made available under the 
% terms of the MIT license (see the LICENSE file) 
% from http://github.com/haythamfayek/MatDL
% Copyright (C) 2016-17 Haytham Fayek.

    p = fieldnames(model);
    for i = 1:numel(p)
        opt.t = opt.t + 1;
        opt.mgrads.(p{i}) = opt.beta1 .* opt.mgrads.(p{i}) + (1 - opt.beta1) .* grads.(p{i});
        opt.vgrads.(p{i}) = opt.beta2 .* opt.vgrads.(p{i}) + (1 - opt.beta2) .* (grads.(p{i}) .^ 2);
        alpha = opt.learningRate .* sqrt(1 - (opt.beta2 .^ opt.t)) ./ (1 - (opt.beta1 .^ opt.t));
        dx = alpha .* opt.mgrads.(p{i}) ./ (sqrt(opt.vgrads.(p{i})) + 1e-8);
        model.(p{i}) = model.(p{i}) - dx;
    end

end

