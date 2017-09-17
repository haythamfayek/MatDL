function [ model, trainLoss, trainAccuracy, valLoss, valAccuracy, opt ] = train( X, Y, XVal, YVal, model, lossfun, opt )
%TRAIN Train a model
%   Inputs:
%       - X: training dataset, of size: number of data points (M) x input dimensions (d)
%       - Y: training labels, of size: number of data points (M) x number of classes (K)
%       - XVal: validation dataset, of size: number of data points (M) x input dimensions (d)
%       - YVal: validations labels, of size: number of data points (M) x number of classes (K)
%       - model: a structure of model weights
%       - lossfun: a function handle to the model
%       - opt: a structure of hyper-parameters
%   Outputs:
%       - model: a structure of trained model weights
%       - trainLoss: training loss history
%       - trainAccuracy: training accuracy history
%       - valLoss: validation loss history
%       - valAccuracy: validation accuracy history
%       - opt: a structure of updated hyper-parameters
%
% This file is part of the MaTDL toolbox and is made available under the 
% terms of the MIT license (see the LICENSE file) 
% from http://github.com/haythamfayek/MatDL
% Copyright (C) 2016-17 Haytham Fayek.

%% Initialize
m = size(X, 1); % Number of Training Examples
mVal = size(XVal, 1);
batchSize = opt.batchSize;
numBatches = fix(m / batchSize);
optim = opt.optim;
maxNorm = opt.maxNorm;

trainLoss = zeros(1, opt.maxEpochs); trainAccuracy = trainLoss;
valLoss = zeros(1, opt.maxEpochs); valAccuracy = valLoss;

%% Main
disp('Start Training ...');
for epoch = 1:opt.maxEpochs
    opt.epoch = epoch;

    %% Train
    rLoss = 0;
    ii = randperm(m);
    for batch = 1:numBatches
        x = X(ii((batch - 1) * batchSize + 1 : batch * batchSize), :, :, :);
        y = Y(ii((batch - 1) * batchSize + 1 : batch * batchSize), :);

        [loss, grads, opt] = lossfun(x, model, y, opt); % Fprop & BackProp
        rLoss = rLoss + loss;

        % Update model
        [model, opt] = optim(model, grads, opt);

        % MaxNorm Constrain
        p = fieldnames(model);
        for i = 1:numel(p)
            if (maxNorm)
                if p{i}(1) == 'w' % only update weights, exclude biases
                    w = model.(p{i});
                    actualNorms = sqrt(sum(w.^2));
                    desiredNorms = actualNorms;
                    desiredNorms(actualNorms > maxNorm) = maxNorm;
                    model.(p{i}) = bsxfun(@times, w, (desiredNorms ./ (actualNorms + eps) ));
                end
            end
        end
    end
    trainLoss(epoch) = gather(rLoss / numBatches);

    %% Evaluate on Training
    rAccuracy = 0;
    ii = randperm(m);
    if numel(ii) > 10000, ii = ii(1:10000); end
    evalSize = 200; % Evaluate using Bigger Batches
    for batch = 1:ceil(numel(ii) / evalSize)
        x = X(ii((batch - 1) * evalSize + 1 : min(numel(ii), batch * evalSize)), :, :, :);
        y = Y(ii((batch - 1) * evalSize + 1 : min(numel(ii), batch * evalSize)), :);
        yp = lossfun(x, model, 0, opt);
        [~, yplabel] = max(yp, [], 2);
        [~, ylabel] = max(y, [], 2);
        rAccuracy = rAccuracy + sum(yplabel == ylabel);
    end
    trainAccuracy(epoch) = gather(rAccuracy / numel(ii));

    %% Evaluate on Validation
    rAccuracy = 0;
    ii = randperm(mVal);
    evalSize = 200; % Evaluate using Bigger Batches
    for batch = 1:ceil(numel(ii) / evalSize)
        x = XVal(ii((batch - 1) * evalSize + 1 : min(numel(ii), batch * evalSize)), :, :, :);
        y = YVal(ii((batch - 1) * evalSize + 1 : min(numel(ii), batch * evalSize)), :);
        yp = lossfun(x, model, 0, opt);
        [~, yplabel] = max(yp, [], 2);
        [~, ylabel] = max(y, [], 2);
        rAccuracy = rAccuracy + sum(yplabel == ylabel);
    end
    valAccuracy(epoch) = gather(rAccuracy / numel(ii));
    valLoss(epoch) = 0; % To Do Later

    %% Display Progress
    disp(['Epoch ' num2str(epoch) '/' num2str(opt.maxEpochs) ' Loss: ' num2str(trainLoss(epoch)) ...
        '. Training Accuracy: ' num2str(100 * trainAccuracy(epoch)) ...
        '%. Validation Accuracy: ' num2str(100 * valAccuracy(epoch)), '%']);

    if (opt.plotProgress)
        subplot(1,2,1);
        plot(trainLoss(1:epoch)); hold on;
        plot(valLoss(1:epoch)); hold off;
        xlabel('Epochs'); title('CE Error'); legend('Training','Validation');

        subplot(1,2,2);
        plot(1 - trainAccuracy(1:epoch)); hold on;
        plot(1 - valAccuracy(1:epoch)); hold off;
        xlabel('Epochs'); title('Classification Error'); legend('Training','Validation');
        drawnow;
    end

    %% Early Stopping
    if (opt.earlyStoppingPatience)
        if (epoch == 1)
            bestModel = model; disp('Best So Far! Checkpoint.');
            bestAccuracy = valAccuracy(epoch); % valLoss(epoch)
            bestEpoch = epoch;
            earlyStoppingCounter = 0;
            bestOpt = opt;
        elseif (valAccuracy(epoch) > bestAccuracy) % valLoss(epoch)
            bestModel = model; disp('Best So Far! Checkpoint.');
            bestAccuracy = valAccuracy(epoch); % valLoss(epoch)
            bestEpoch = epoch;
            earlyStoppingCounter = 0;
            bestOpt = opt;
        else
            earlyStoppingCounter = earlyStoppingCounter + 1;
        end
        if (earlyStoppingCounter >= opt.earlyStoppingPatience) || (epoch == opt.maxEpochs)
            model = bestModel;
            opt.bestEpoch = bestEpoch;
            opt.bestOpt = bestOpt;
            break;
        end
    end
        
    %% Anneal Learning Rate
    if strcmp(opt.learningDecaySchedule,'step')
        if (mod(epoch, opt.learningDecayRateStep) == 0), opt.learningRate = opt.learningRate * opt.learningDecayRate; end

    elseif strcmp(opt.learningDecaySchedule,'stepsave')
        if (mod(epoch, opt.learningDecayRateStep) == 0)
            opt.learningRate = opt.learningRate * opt.learningDecayRate;
            model = bestModel; % Revert to best model
            lR = opt.learningRate; ep = opt.epoch; % Save the variable settings
            opt = bestOpt; % Revert to best settings
            opt.learningRate = lR; opt.epoch = ep; % Load back the variable settings
        end

    elseif strcmp(opt.learningDecaySchedule,'t/T')
        opt.learningRate = opt.learningRate / (1 + epoch / opt.maxEpochs);

    elseif strcmp(opt.learningDecaySchedule,'exp')
        opt.learningRate = opt.learningRate * opt.learningDecayRate;
    end
    
end % Reached Max Epochs

end
