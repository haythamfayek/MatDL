function [model, opt] = init_five_convnet(N, K, layersSize, opt)

    weightScale = 0.01;
    biasScale = 0;

    C = N(1); H = N(2); W = N(3); % Input shape
    NL1 = layersSize(1); NL2 = layersSize(2); NL3 = layersSize(3); NL4 = layersSize(4);

    model.w1 = randn(NL1, C, 3, 3) * weightScale;
    model.b1 = randn(1, NL1) * biasScale;
    model.w2 = randn(NL2, NL1, 4, 4) * weightScale;
    model.b2 = randn(1, NL2) * biasScale;
    model.w3 = randn(5 * 5 * NL2, NL3) * weightScale;
    model.b3 = randn(1, NL3) * biasScale;
    model.w4 = randn(NL3, NL4) * weightScale;
    model.b4 = randn(1, NL4) * biasScale;
    model.w5 = randn(NL4, K) * weightScale;
    model.b5 = randn(1, K) * biasScale;
    
    model.gamma1 = ones(1, NL1);
    model.beta1 = zeros(1, NL1);
    model.gamma2 = ones(1, NL2);
    model.beta2 = zeros(1, NL2);
    model.gamma3 = ones(1, NL3);
    model.beta3 = zeros(1, NL3);
    model.gamma4 = ones(1, NL4);
    model.beta4 = zeros(1, NL4);
    
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

end