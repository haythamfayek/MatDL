function [dx, dw, db] = conv_relu_pool_backward(dout, cache)
    convCache = cache.convCache;
    reluCache = cache.reluCache;
    poolCache = cache.poolCache;
    ds = max_pool_backward(dout, poolCache);
    da = relu_backward(ds, reluCache);
    [dx, dw, db] = conv_backward(da, convCache);
end
