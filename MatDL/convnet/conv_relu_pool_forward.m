function [out, cache] = conv_relu_pool_forward(x, w, b, convParam, poolParam)
    [a, convCache] = conv_forward(x, w, b, convParam);
    [s, reluCache] = relu_forward(a);
    [out, poolCache] = max_pool_forward(s, poolParam);
    cache.convCache = convCache; cache.reluCache = reluCache; 
    cache.poolCache = poolCache;
end
