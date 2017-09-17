function [out, grads, opt] = lstm_rnn(X, model, y, opt)

    weightDecay = opt.weightDecay;
    dropout = opt.dropout;
    extractFeature = opt.extractFeature;
    computeDX = opt.computeDX;

    WLSTM = model.WLSTM;
    WLSTM2 = model.WLSTM2;
    wy = model.wy; by = model.by;

    H = size(WLSTM, 2) / 4;
    h0 = zeros(size(X,1), H);
    c0 = zeros(size(X,1), H);

    dropoutParam.p = dropout;
    if gather(y == 0), dropoutParam.mode = 'test'; else dropoutParam.mode = 'train'; end
    dropoutParam.useGPU = opt.useGPU;

    [a1, ~, ~, cache1] = lstm_forward(X, h0, WLSTM, c0);
    [d1, cached1] = dropout_forward(a1, dropoutParam);
    [a2, ~, ~, cache2] = lstm_forward(d1, h0, WLSTM2, c0);
    [d2, cached2] = dropout_forward(a2, dropoutParam);
    [scores, cache3] = forward(d2(:, :, end), wy, by);

    if (extractFeature), out = scores; grads = 0; return; end

    yp = softmax_forward(scores);

    if gather(y == 0), out = yp; grads = 0; return; end

    [dataLoss, dscores] = softmax_backward(yp, y);
    [dd2, dwy, dby] = backward(dscores, cache3);
    ddd2 = zeros([size(dd2), size(X, 3)]); ddd2(:, :, end) = dd2; % No GPU
    da2 = dropout_backward(ddd2, cached2);
    [dd1, dWLSTM2, ~, ~] = lstm_backward(da2, cache2, 0, 0);
    da1 = dropout_backward(dd1, cached1);
    [dX, dWLSTM, ~, ~] = lstm_backward(da1, cache1, 0, 0);

    if (computeDX), out = dX; grads = 0; return; end

    grads.WLSTM = dWLSTM; grads.WLSTM2 = dWLSTM2; grads.wy = dwy; grads.by =dby;

    W = {'WLSTM', 'WLSTM2', 'wy', 'by'};
    for i = 1:numel(W)
        grads.(W{i})(grads.(W{i}) < -5) = -5;
        grads.(W{i})(grads.(W{i}) > 5) = 5;
    end

    regLoss = 0;
    if (weightDecay)
        W = {'WLSTM', 'WLSTM2', 'wy'};
        for i = 1:numel(W)
            w = model.(W{i});
            regLoss = regLoss + 0.5 * weightDecay * sum(sum(w.^2));
            grads.(W{i}) = grads.(W{i}) + weightDecay * w;
        end
    end
    out = dataLoss + regLoss;

end
