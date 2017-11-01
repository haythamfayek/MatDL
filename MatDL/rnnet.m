% A complete example of a recurrent neural network
%% Init
clear all
addpath(genpath('../MatDL'));

%% Load data
load('../Data/mnist_uint8.mat');
X = double(reshape(train_x',28,28,60000))/255; X = permute(X, [3 2 1]);
XVal = double(reshape(test_x',28,28,10000))/255; XVal = permute(XVal, [3 2 1]);
Y = double(train_y);
YVal = double(test_y);

%% Initialize model
opt = struct;

[model, opt] = init_two_rnn(28, 10, [10 10], opt);

%% Hyper-parameters
opt.batchSize = 100;

opt.optim = @rmsprop;
% opt.beta1 = 0.9; opt.beta2 = 0.999; opt.t = 0; opt.mgrads = opt.vgrads;
opt.rmspropDecay = 0.99;
% opt.initialMomentum = 0.5; opt.switchEpochMomentum = 1; opt.finalMomentum = 0.9;
opt.learningRate = 0.001;
opt.learningDecaySchedule = 'exp'; % 'no_decay', 't/T', 'step'
opt.learningDecayRate = 1;
% opt.learningDecayRateStep = 5;

opt.dropout = 1;
opt.weightDecay = false;
opt.maxNorm = false;

opt.maxEpochs = 1;
opt.earlyStoppingPatience = 5;
opt.valFreq = 100;

opt.plotProgress = false;
opt.extractFeature = false;
opt.computeDX = false;

opt.useGPU = false;
if (opt.useGPU) % Copy data, dropout, model, vgrads, BNParams
    X = gpuArray(X); Y = gpuArray(Y); XVal = gpuArray(XVal); YVal = gpuArray(YVal); 
    opt.dropout = gpuArray(opt.dropout);
    p = fieldnames(model);
    for i = 1:numel(p), model.(p{i}) = gpuArray(model.(p{i})); opt.vgrads.(p{i}) = gpuArray(opt.vgrads.(p{i})); end
    p = fieldnames(opt);
    for i = 1:numel(p), if (strfind(p{i},'bnParam')), opt.(p{i}).runningMean = gpuArray(opt.(p{i}).runningMean); opt.(p{i}).runningVar = gpuArray(opt.(p{i}).runningVar); end; end
end

%% Gradient check
x = X(1:100, :, :);
y = Y(1:100, :, :);
maxRelError = gradcheck(@two_rnn, x, model, y, opt, 5);

%% Train
[model, trainLoss, trainAccuracy, valLoss, valAccuracy, opt] = train( X, Y, XVal, YVal, model, @two_rnn, opt );

%% Predict
[yplabel, confidence, classes, classConfidences, yp] = predict(XVal, @two_rnn, model, opt)