function [out, grads, opt] = x_nn_bn(X, model, y, opt)

    weightDecay = opt.weightDecay; 
    dropout = opt.dropout;
    extractFeature = opt.extractFeature;
    computeDX = opt.computeDX;
    p = fieldnames(model);
    numLayers = numel(opt.layersSize);
    
    a = cell(numLayers, 1); cache = cell(numLayers, 1);
    d = cell(numLayers, 1); cached = cell(numLayers, 1);
    da = cell(numLayers, 1); dd = cell(numLayers, 1);
    
    dropoutParam.p = dropout;
    if gather(y == 0), dropoutParam.mode = 'test'; else dropoutParam.mode = 'train'; end
    dropoutParam.useGPU = opt.useGPU;
    
    for i = 1:numLayers
        if gather(y == 0),  opt.(['bnParam', num2str(i)]).mode = 'test';
        else opt.(['bnParam', num2str(i)]).mode = 'train'; end     
    end

    [a{1}, opt.(['bnParam', num2str(1)]), cache{1}] = forward_bn_relu(X, model.w1, model.b1, model.gamma1, model.beta1, opt.bnParam1);
    [d{1}, cached{1}] = dropout_forward(a{1}, dropoutParam);
    
    for i = 2:numLayers
        [a{i}, opt.(['bnParam', num2str(i)]), cache{i}] = forward_bn_relu(d{i - 1}, model.(['w', num2str(i)]), model.(['b', num2str(i)]), model.(['gamma', num2str(i)]), model.(['beta', num2str(i)]), opt.(['bnParam', num2str(i)]));
        [d{i}, cached{i}] = dropout_forward(a{i}, dropoutParam);
    end
    
    [scores, cache2] = forward(d{numLayers}, model.(['w', num2str(numLayers + 1)]), model.(['b', num2str(numLayers + 1)]));
    
    if (extractFeature), out = scores; grads = 0; return; end
    
    yp = softmax_forward(scores);
    
    if gather(y == 0), out = yp; grads = 0; return; end
    
    [dataLoss, dscores] = softmax_backward(yp, y);
    [dd{numLayers}, grads.(['w', num2str(numLayers + 1)]), grads.(['b', num2str(numLayers + 1)])] = backward(dscores, cache2);
    
    for i = numLayers:-1:2
        da{i} = dropout_backward(dd{i}, cached{i});
        [dd{i - 1}, grads.(['w', num2str(i)]), grads.(['b', num2str(i)]), grads.(['gamma', num2str(i)]), grads.(['beta', num2str(i)])] = backward_bn_relu(da{i}, cache{i});
    end
    
    da{1} = dropout_backward(dd{1}, cached{1});
    [dX, grads.w1, grads.b1, grads.gamma1, grads.beta1] = backward_bn_relu(da{1}, cache{1});
    
    if (computeDX), out = dX; grads = 0; return; end
    
    regLoss = 0;
    if (weightDecay)
        iW = strncmpi(p, 'w', 1);
        W = p(iW);
        for i = 1:numel(W)
            w = model.(W{i});
            regLoss = regLoss + 0.5 * weightDecay * sum(sum(w.^2));
            grads.(W{i}) = grads.(W{i}) + weightDecay * w;
        end
    end
    out = dataLoss + regLoss;

end