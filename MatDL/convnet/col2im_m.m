function xPadded = col2im_m(cols, M, C, H, W, filterHeight, filterWidth, pad, stride)
%COL2IM_M Reshape columns into image blocks
%   Inputs:
%       - cols: columns to be reshaped into image blocks
%       - M: number of images
%       - C: number of channels
%       - H: height of each image
%       - W: width of each image
%       - filterHeight: layer filter height
%       - filterWidth: layer filter width
%       - pad: layer padding
%       - stride: layer convolution stride
%   Outputs:
%       - xPadded: image blocks
%
% This file is part of the MaTDL toolbox and is made available under the 
% terms of the MIT license (see the LICENSE file) 
% from http://github.com/haythamfayek/MatDL
% Copyright (C) 2016-17 Haytham Fayek.

    HH = (H + 2 * pad - filterHeight) / stride + 1;
    WW = (W + 2 * pad - filterWidth) / stride + 1;
    
    xPadded = zeros(M, C, H + 2 * pad, W + 2 * pad);
    
    for c = 0:C-1
        for ii = 0:filterHeight-1
            for jj = 0:filterWidth-1
                row = c * filterWidth * filterHeight + ii * filterWidth + jj;
                for yy = 0:HH-1
                    for xx = 0:WW-1
                        for i = 0:M-1
                            col = yy * WW * M + xx * M + i;
                            xPadded(i+1, c+1, stride*yy+ii+1, stride*xx+jj+1) = xPadded(i+1, c+1, stride*yy+ii+1, stride*xx+jj+1) + cols(row+1, col+1);
                        end
                    end
                end
            end
        end
    end
    
    if pad > 0
        xPadded = xPadded(:, :, (pad + 1 : end - pad), (pad + 1 : end - pad)); 
    end
    
end