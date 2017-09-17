function [out, grads, opt] = four_convnet(X, model, y, opt)

    weightDecay = opt.weightDecay; 
    dropout = opt.dropout;
    extractFeature = opt.extractFeature;
    computeDX = opt.computeDX;

    w1 = model.w1; b1 = model.b1;
    w2 = model.w2; b2 = model.b2;
    w3 = model.w3; b3 = model.b3;
    w4 = model.w4; b4 = model.b4;
    
    convParam.stride = 1; convParam.pad = (size(w1,3) - 1) / 2;
    poolParam.stride = 2; poolParam.poolHeight = 2; poolParam.poolWidth = 2;
    dropoutParam.p = dropout;
    if gather(y == 0), dropoutParam.mode = 'test'; else dropoutParam.mode = 'train'; end
    dropoutParam.useGPU = opt.useGPU;
    poolParam.useGPU = opt.useGPU;
    convParam.useGPU = opt.useGPU;
    
    [a1, cache1] = conv_relu_pool_forward(X, w1, b1, convParam, poolParam);
    [a2, cache2] = conv_relu_pool_forward(a1, w2, b2, convParam, poolParam);
    [a3, cache3] = affine_relu_forward(a2, w3, b3);
    [d3, cached3] = dropout_forward(a3, dropoutParam);
    [scores, cache4] = affine_forward(d3, w4, b4);
    
    if (extractFeature), out = scores; grads = 0; return; end
    
    yp = softmax_forward(scores);
    
    if gather(y == 0), out = yp; grads = 0; return; end
    
    [dataLoss, dscores] = softmax_backward(yp, y);
    [dd3, dw4, db4] = affine_backward(dscores, cache4);
    da3 = dropout_backward(dd3, cached3);
    [da2, dw3, db3] = affine_relu_backward(da3, cache3);
    [da1, dw2, db2] = conv_relu_pool_backward(da2, cache2);
    [dX, dw1, db1] = conv_relu_pool_backward(da1, cache1);
    
    if (computeDX), out = dX; grads = 0; return; end
    
    grads.w1 = dw1; grads.b1 = db1;
    grads.w2 = dw2; grads.b2 = db2;
    grads.w3 = dw3; grads.b3 = db3;
    grads.w4 = dw4; grads.b4 = db4;
    
    regLoss = 0;
    if (weightDecay)
        W = {'w1', 'w2', 'w3', 'w4'};
        for i = 1:numel(W)
            w = model.(W{i});
            regLoss = regLoss + 0.5 * weightDecay * sum(sum(sum(sum(w.^2))));
            grads.(W{i}) = grads.(W{i}) + weightDecay * w;
        end
    end
    out = dataLoss + regLoss;

end