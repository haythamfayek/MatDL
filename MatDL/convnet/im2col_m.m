function cols = im2col_m(x, filterHeight, filterWidth, pad, stride)
%IM2COL_M Reshape image blocks into columns
%   Inputs:
%       - x: image blocks
%       - filterHeight: layer filter height
%       - filterWidth: layer filter width
%       - pad: layer padding
%       - stride: layer convolution stride
%   Outputs:
%       - cols: columns
%
% This file is part of the MaTDL toolbox and is made available under the 
% terms of the MIT license (see the LICENSE file) 
% from http://github.com/haythamfayek/MatDL
% Copyright (C) 2016-17 Haytham Fayek.

    [M, C, H, W] = size(x);
    HH = (H + 2 * pad - filterHeight) / stride + 1;
    WW = (W + 2 * pad - filterWidth) / stride + 1;
    
    xPadded = padarray(x, [0, 0, pad, pad]);
    
    cols = zeros(C * filterHeight * filterWidth, M * HH * WW);
    
    for c = 0:C-1
        for yy = 0:HH-1
            for xx = 0:WW-1
                for ii = 0:filterHeight-1
                    for jj = 0:filterWidth-1
                        row = c * filterWidth * filterHeight + ii * filterWidth + jj; % Changed Height to Width
                        for i = 0:M-1
                            col =  yy * WW * M + xx * M + i;
                            cols(row+1, col+1) = xPadded(i+1, c+1, stride*yy+ii+1, stride*xx+jj+1);
                        end
                    end
                end
            end
        end
    end
    
end