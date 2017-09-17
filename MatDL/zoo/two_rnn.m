function [out, grads, opt] = two_rnn(X, model, y, opt)

    weightDecay = opt.weightDecay;
    dropout = opt.dropout;
    extractFeature = opt.extractFeature;
    computeDX = opt.computeDX;

    wx1 = model.wx1; wh1 = model.wh1; b1 = model.b1;
    wy = model.wy; by = model.by;

    dropoutParam.p = dropout;
    if gather(y == 0), dropoutParam.mode = 'test'; else dropoutParam.mode = 'train'; end
    dropoutParam.useGPU = opt.useGPU;

    hprev = zeros([size(X, 1), size(wh1, 1)]);
    [a1, cache1] = rnn_forward(X, hprev, wx1, wh1, b1); % No GPU
    [d1, cached1] = dropout_forward(a1, dropoutParam);
    [scores, cache2] = forward(d1(:, :, end), wy, by); % Only last one

    if (extractFeature), out = scores; grads = 0; return; end

    yp = softmax_forward(scores);

    if gather(y == 0), out = yp; grads = 0; return; end

    [dataLoss, dscores] = softmax_backward(yp, y);
    [dd1, dwy, dby] = backward(dscores, cache2); % No GPU
    ddd1 = zeros([size(dd1), size(X, 3)]); ddd1(:, :, end) = dd1; % No GPU
    da1 = dropout_backward(ddd1, cached1);
    [dX, dwx1, dwh1, db1] = rnn_backward(da1, cache1);

    if (computeDX), out = dX; grads = 0; return; end

    grads.wx1 = dwx1; grads.wh1 = dwh1; grads.b1 = db1;
    grads.wy = dwy; grads.by = dby;

    regLoss = 0;
    if (weightDecay)
        W = {'wx1', 'wh1', 'wy'};
        for i = 1:numel(W)
            w = model.(W{i});
            regLoss = regLoss + 0.5 * weightDecay * sum(sum(w.^2));
            grads.(W{i}) = grads.(W{i}) + weightDecay * w;
        end
    end
    out = dataLoss + regLoss;

end
