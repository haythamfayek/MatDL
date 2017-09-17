function [dx] = dropout_backward(dout, cache)
%DROPOUT_BACKWARD Compute gradients for a dropout layer
%   Inputs:
%       - dout: upstream derivatives, of size: batch size (m) x layer size (l)
%       - cache: a structure of:
%           mask: mask used to drop units, of size: batch size (m) x layer size (l)
%           param: hyper-parameters of layer
%   Outputs:
%       - dx: gradients w.r.t x, of size: size(x)
%
% This file is part of the MaTDL toolbox and is made available under the 
% terms of the MIT license (see the LICENSE file) 
% from http://github.com/haythamfayek/MatDL
% Copyright (C) 2016-17 Haytham Fayek.

    param = cache.param; mask = cache.mask;
    mode = param.mode;
    
    if strcmp(mode, 'train')
        dx = dout .* mask;
        
    elseif strcmp(mode, 'test')
        dx = dout;
        
    else
        error('DropOut Malfunction');
    end
    
end