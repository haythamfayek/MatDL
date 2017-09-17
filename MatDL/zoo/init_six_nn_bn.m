function [model, opt] = init_six_nn_bn(N, K, layersSize, opt)

    bias_scale = 0;

    NL1 = layersSize(1); NL2 = layersSize(2);
    NL3 = layersSize(3); NL4 = layersSize(4);
    NL5 = layersSize(5);

    model.w1 = randn(N, NL1) * sqrt(2/N);
    model.b1 = repmat(0.01, 1, NL1);
    model.w2 = randn(NL1, NL2) * sqrt(2/NL1);
    model.b2 = repmat(0.01, 1, NL2);
    model.w3 = randn(NL2, NL3) * sqrt(2/NL2);
    model.b3 = repmat(0.01, 1, NL3);
    model.w4 = randn(NL3, NL4) * sqrt(2/NL3);
    model.b4 = repmat(0.01, 1, NL4);
    model.w5 = randn(NL4, NL5) * sqrt(2/NL4);
    model.b5 = repmat(0.01, 1, NL5);
    % The following layers are not ReLUs, so have different initialization
    model.w6 = randn(NL5, K) * sqrt(1/NL5);
    model.b6 = randn(1, K) * bias_scale;
    
    model.gamma1 = ones(1, NL1);
    model.beta1 = zeros(1, NL1);
    model.gamma2 = ones(1, NL2);
    model.beta2 = zeros(1, NL2);
    model.gamma3 = ones(1, NL3);
    model.beta3 = zeros(1, NL3);
    model.gamma4 = ones(1, NL4);
    model.beta4 = zeros(1, NL4);
    model.gamma5 = ones(1, NL5);
    model.beta5 = zeros(1, NL5);
  
    p = fieldnames(model);
    for i = 1:numel(p)
        opt.vgrads.(p{i}) = zeros(size(model.(p{i})));
    end
    
    opt.bnParam1.runningMean = zeros(1, NL1);
    opt.bnParam1.runningVar = zeros(1, NL1);
    opt.bnParam2.runningMean = zeros(1, NL2);
    opt.bnParam2.runningVar = zeros(1, NL2);
    opt.bnParam3.runningMean = zeros(1, NL3);
    opt.bnParam3.runningVar = zeros(1, NL3);
    opt.bnParam4.runningMean = zeros(1, NL4);
    opt.bnParam4.runningVar = zeros(1, NL4);
    opt.bnParam5.runningMean = zeros(1, NL5);
    opt.bnParam5.runningVar = zeros(1, NL5);

end