# MatDL

[![status](http://joss.theoj.org/papers/fa33e01459843ac8a5b781b1bd0c3250/status.svg)](http://joss.theoj.org/papers/fa33e01459843ac8a5b781b1bd0c3250)
[![DOI](https://zenodo.org/badge/103798084.svg)](https://zenodo.org/badge/latestdoi/103798084)

*MatDL* is an open-source lightweight deep learning library native in MATLAB that implements some most commonly used deep learning algorithms. 
The library comprises functions that implement the following: (1) basic building blocks of modern neural networks such as affine transformations, convolutions, nonlinear operations, dropout, batch normalization, etc.; (2) popular architectures such as deep neural networks (DNNs), convolutional neural networks (ConvNets), and recurrent neural networks (RNNs) and their variant, the long short-term memory (LSTM) RNNs; (3) optimizers such stochastic gradient descent (SGD), RMSProp and ADAM; as well as (4) boilerplate functions for training, gradients checking, etc.

## Installation

- Add *MatDL* to your path:
```matlab
addpath(genpath('MatDL'));
```

- Compile the C MEX files using the `Makefile` or using the following:
```matlab
cd MatDL/convnet; mex col2im_mex.c; mex im2col_mex.c; cd ..; cd ..;
```

## Project Layout

`common/`: Basic building blocks of neural networks such as nonlinear functions, etc.

`convnet/`: ConvNet specific functions such as convolutional layers, max pooling layers, etc.

`nn/`: NN specific functions.

`optim/`: Optimization algorithms such as SGD, RMSProp, ADAM, etc.

`rnn/`: RNN and LSTM functions.

`train/`: Functions for gradients checking, training, and prediction.

`zoo/`: Samples of various models definitions and initializations.

## Usage

This is a sample complete minimum working example:
(Examples: [DNN](MatDL/nnet.m) (Below), [ConvNet](MatDL/convnet.m), [RNN](MatDL/rnnet.m))

```matlab
% A complete minimum working example.
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
```

## Contributing

Contributions are highly welcome!

If you wish to contribute, follow these steps:
- Create a personal fork of the MatDL GitHub repository.
- Make your changes in a branch named other than `master`.
- Follow the same coding and documentation style for consistency.
- Make sure that your changes do not break any of the existing functionality.
- Submit a pull request.

Please use Github issues to report bugs or request features.

## Citation

If you use this library in your research, please cite:

`Fayek, (2017). MatDL: A Lightweight Deep Learning Library in MATLAB. Journal of Open Source Software, 2(19), 413, doi:10.21105/joss.00413`

```
@article{Fayek2017,
    author       = {Haytham M. Fayek},
    title        = {{MatDL}: A Lightweight Deep Learning Library in {MATLAB}},
    journal      = {Journal of Open Source Software},
    year         = {2017},
    month        = {nov},
    volume       = {2},
    number       = {19},
    doi          = {10.21105/joss.00413},
    url          = {https://doi.org/10.21105/joss.00413},
    publisher    = {The Open Journal},
}
```

## References

*MatDL* was inspired by Stanford's CS231n and Torch, and is conceptually similar to Keras and Lasagne.
Torch, keras and Lasagne are more suited for large-scale experiments.

## License

MIT
