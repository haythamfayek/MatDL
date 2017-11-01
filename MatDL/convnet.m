% A complete example of a convolutional neural network
%% Init
clear all
addpath(genpath('../MatDL'));

%% Load data
load('../Data/mnist_uint8.mat');
X = double(reshape(train_x',1,28,28,60000))/255; X = permute(X, [4 1 3 2]);
XVal = double(reshape(test_x',1,28,28,10000))/255; XVal = permute(XVal, [4 1 3 2]);
Y = double(train_y);
YVal = double(test_y);

%% Initialize model
opt = struct;
[model, opt] = init_five_convnet_stride([1, 28, 28], 10, [16, 32, 128, 256, 256], opt);

%% Hyper-parameters
opt.useGPU = false;

opt.batchSize = 100;

opt.optim = @rmsprop; % 'sgd', 'nestrov'
opt.rmspropDecay = 0.99;
% opt.initialMomentum = 0.5; opt.switchEpochMomentum = 1; opt.finalMomentum = 0.9;
opt.learningRate = 0.01;
opt.learningDecaySchedule = 'exp'; % 'no_decay', 't/T', 'step'
opt.learningDecayRate = 0.8;
% opt.learningDecayRateStep = 5;

opt.dropout = 1;
opt.weightDecay = 0;
opt.maxNorm = 0;

opt.maxEpochs = 1;
opt.earlyStoppingPatience = 10;

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
x = X(1:10,:,:,:);
y = Y(1:10,:);
maxRelError = gradcheck(@five_convnet_stride, x, model, y, opt, 8);

%% Train
[model, trainLoss, trainAccuracy, valLoss, valAccuracy, opt] = train( X, Y, XVal, YVal, model, @five_convnet_stride, opt );

%% Predict
[yplabel, confidence, classes, classConfidences, yp] = predict(XVal, @five_convnet_stride, model, opt)