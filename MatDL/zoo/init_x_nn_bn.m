function [model, opt] = init_x_nn_bn(N, K, layersSize, opt)

    bias_scale = 0;
    layersSize = [N, layersSize];
    for i = 1:numel(layersSize)-1
        model.(['w', num2str(i)]) = randn(layersSize(i), layersSize(i+1)) * sqrt(2/layersSize(i));
        model.(['b', num2str(i)]) = repmat(0.01, 1, layersSize(i+1));
    end
    % The following layers are not ReLUs, so have different initialization
    model.(['w', num2str(numel(layersSize))]) = randn(layersSize(end), K) * sqrt(1/layersSize(end));
    model.(['b', num2str(numel(layersSize))]) = randn(1, K) * bias_scale;
    
    layersSize = layersSize(2:end);
    for i = 1:numel(layersSize)
        model.(['gamma', num2str(i)]) = ones(1, layersSize(i));
        model.(['beta', num2str(i)]) = zeros(1, layersSize(i));
    end
  
    p = fieldnames(model);
    for i = 1:numel(p)
        opt.vgrads.(p{i}) = zeros(size(model.(p{i})));
    end
    
    for i = 1:numel(layersSize)
        opt.(['bnParam', num2str(i)]).runningMean = zeros(1, layersSize(i));
        opt.(['bnParam', num2str(i)]).runningVar = zeros(1, layersSize(i));
    end
    
    opt.layersSize = layersSize;

end