function [model, opt] = init_two_rnn(N, K, layers_size, opt)

    weightScale = 0.01;
    biasScale = 0;

    NL1 = layers_size(1);

    model.wx1 = randn(N, NL1) * weightScale;
    model.wh1 = randn(NL1, NL1) * weightScale;
    model.b1 = randn(1, NL1) * biasScale;
    model.wy = randn(NL1, K) * weightScale;
    model.by = randn(1, K) * biasScale;
    
    p = fieldnames(model);
    for i = 1:numel(p)
        opt.vgrads.(p{i}) = zeros(size(model.(p{i})));
    end

end