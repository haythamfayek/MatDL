function [ maxRelError ] = gradcheck( lossfun, x, model, y, opt, numChecks  )
%GRADCHECK Verify analytic gradients vs numerical gradients
%   Inputs:
%       - lossfun: a function handle to the model
%       - x: input to model, of size: batch size (m) x input dimensions (d)
%       - model: a structure of model weights
%       - y: input labels, of size: batch size (m) x number of classes (K)
%       - opt: a structure of hyper-parameters
%       - numChecks: number of gradient evaluations for each model variable
%   Outputs:
%       - maxRelError: maximum relative error found across all model variables
%
% This file is part of the MaTDL toolbox and is made available under the 
% terms of the MIT license (see the LICENSE file) 
% from http://github.com/haythamfayek/MatDL
% Copyright (C) 2016-17 Haytham Fayek.

    h = 1e-5;
    maxRelError = 0;
    
    rng(0);
    [loss, gradAnalytic] = lossfun(x, model, y, opt);
    disp(['Loss is ' num2str(loss)])
    
    p = fieldnames(model);
    for i = 1:numel(p)
        disp(p{i})
        mn = numel(model.(p{i}));
        imn = (randperm(mn, numChecks));
        for j = 1:numChecks
            oldVal = model.(p{i})(imn(j));

            model.(p{i})(imn(j)) = oldVal + h; rng(0);
            fxph = lossfun(x, model, y, opt);

            model.(p{i})(imn(j)) = oldVal - h; rng(0);
            fxmh = lossfun(x, model, y, opt);

            model.(p{i})(imn(j)) = oldVal; % reset

            gN = (fxph - fxmh) / (2 * h);
            gA = gradAnalytic.(p{i})(imn(j));
            relError = abs(gN - gA) / max(1e-8, (abs(gN) + abs(gA)));

            disp(['Numerical: ' num2str(gN) '. Analytic: ' num2str(gA) '. Relative Error: ' num2str(relError)]);

            if relError > maxRelError, maxRelError = relError; end
        end
    end
end