% A complete example of a deep neural network
%% Init
clear all
addpath(genpath('../MatDL'));

%% Load data
load('../Data/mnist_uint8.mat'); % Replace with your data file
X = double(train_x)/255; Y = double(train_y);
XVal = double(test_x)/255; YVal =  double(test_y);

rng(0);

%% Initialize model
opt = struct;
[model, opt] = init_six_nn_bn(784, 10, [100, 100, 100, 100, 100], opt);

%% Hyper-parameters
opt.batchSize = 100;

opt.optim = @rmsprop;
% opt.beta1 = 0.9; opt.beta2 = 0.999; opt.t = 0; opt.mgrads = opt.vgrads;
opt.rmspropDecay = 0.99;
% opt.initialMomentum = 0.5; opt.switchEpochMomentum = 1; opt.finalMomentum = 0.9;
opt.learningRate = 0.01;
opt.learningDecaySchedule = 'stepsave'; % 'no_decay', 't/T', 'step'
opt.learningDecayRate = 0.5;
opt.learningDecayRateStep = 5;

opt.dropout = 0.5;
opt.weightDecay = false;
opt.maxNorm = false;

opt.maxEpochs = 100;
opt.earlyStoppingPatience = 20;
opt.valFreq = 100;

opt.plotProgress = true;
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
x = X(1:100,:);
y = Y(1:100,:);
maxRelError = gradcheck(@six_nn_bn, x, model, y, opt, 10);

%% Train
[model, trainLoss, trainAccuracy, valLoss, valAccuracy, opt] = train(X, Y, XVal, YVal, model, @six_nn_bn, opt);

%% Predict
[yplabel, confidence, classes, classConfidences, yp] = predict(XVal, @six_nn_bn, model, opt)