function [model, opt] = rmsprop(model, grads, opt)
%RMSPROP Do an RMS prop update step
%   Inputs:
%       - model: a structure of model parameters to update
%       - grads: a structure of gradients of model parameters
%       - opt: a structure of hyper-parameters and cache for rmsprop
%   Outputs:
%       - model: a structure of updated model parameters
%       - opt: a structure of updated hyper-parameters and cache for rmsprop
%
% This file is part of the MaTDL toolbox and is made available under the 
% terms of the MIT license (see the LICENSE file) 
% from http://github.com/haythamfayek/MatDL
% Copyright (C) 2016-17 Haytham Fayek.

    p = fieldnames(model);
    for i = 1:numel(p)
        rmspropDecay = opt.rmspropDecay;
        opt.vgrads.(p{i}) = opt.vgrads.(p{i}) * rmspropDecay + (1.0 - rmspropDecay) * (grads.(p{i}).^2);
        dx = -(opt.learningRate * grads.(p{i})) ./ sqrt(opt.vgrads.(p{i}) + 1e-8);
        model.(p{i}) = model.(p{i}) + dx;
    end

end

