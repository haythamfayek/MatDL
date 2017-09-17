function [model, opt] = init_four_convnet(N, K, filterSize, layersSize, opt)

    weightScale = 0.01;
    biasScale = 0;

    C = N(1); H = N(2); W = N(3); % Input shape
    NL1 = layersSize(1); NL2 = layersSize(2); NL3 = layersSize(3);

    model.w1 = randn(NL1, C, filterSize(1), filterSize(1)) * weightScale;
    model.b1 = randn(1, NL1) * biasScale;
    model.w2 = randn(NL2, NL1, filterSize(2), filterSize(2)) * weightScale;
    model.b2 = randn(1, NL2) * biasScale;
    model.w3 = randn(H * W * NL2 / 16, NL3) * weightScale;
    model.b3 = randn(1, NL3) * biasScale;
    model.w4 = randn(NL3, K) * weightScale;
    model.b4 = randn(1, K) * biasScale;
    
    p = fieldnames(model);
    for i = 1:numel(p)
        opt.vgrads.(p{i}) = zeros(size(model.(p{i})));
    end

end