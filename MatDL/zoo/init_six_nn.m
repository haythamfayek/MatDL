function [model, opt] = init_six_nn(N, K, layersSize, opt)

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
    
    p = fieldnames(model);
    for i = 1:numel(p)
        opt.vgrads.(p{i}) = zeros(size(model.(p{i})));
    end

end