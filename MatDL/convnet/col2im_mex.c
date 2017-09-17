/*
 * Reshape columns into image blocks
 *
 * This file is part of the MaTDL toolbox and is made available under the 
 * terms of the MIT license (see the LICENSE file) 
 * from http://github.com/haythamfayek/MatDL
 * Copyright (C) 2016-17 Haytham Fayek.
 */

#include <math.h>
#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
    
    #define x_padded_out plhs[0]
    #define cols_in prhs[0]
    #define M_in prhs[1]
    #define C_in prhs[2]
    #define H_in prhs[3]
    #define W_in prhs[4]
    #define HH_in prhs[5]
    #define WW_in prhs[6]
    #define filter_height_in prhs[7]
    #define filter_width_in prhs[8]
    #define pad_in prhs[9]
    #define stride_in prhs[10]
    #define x_pad_in prhs[11]
    
    double *cols, *x_pad;
    int M, C, H, W, HH, WW, filter_height, filter_width, pad, stride;
    cols = mxGetPr(cols_in);
    M = mxGetScalar(M_in);
    C = mxGetScalar(C_in);
    H = mxGetScalar(H_in);
    W = mxGetScalar(W_in);
    HH = mxGetScalar(HH_in);
    WW = mxGetScalar(WW_in);
    filter_height = mxGetScalar(filter_height_in);
    filter_width = mxGetScalar(filter_width_in);
    pad = mxGetScalar(pad_in);
    stride = mxGetScalar(stride_in);
    /* x_pad = mxGetPr(x_pad_in); */
    
    double *x_padded;
    x_padded_out = mxDuplicateArray(x_pad_in);
    x_padded = mxGetPr(x_padded_out);
    
    int row, col;
    int c, ii, jj, yy, xx, i; 
    
    for(c = 0; c < C; c++){
        for(ii = 0; ii < filter_height; ii++){
            for(jj = 0; jj < filter_width; jj++){
                row = (c * filter_width * filter_height) + (ii * filter_width) + jj; /* Changed Height to Width */
                for(yy = 0; yy < HH; yy++){
                    for(xx = 0; xx < WW; xx++){
                        for(i = 0; i < M; i++){
                            col = (yy * WW * M) + (xx * M) + i;
                            x_padded[i + (M * (c + (C * (((stride * yy) + ii) + ((H + 2 * pad) * ((stride * xx) + jj))))))] += cols[row +  ((C * filter_height * filter_width) * col)];
                        }
                    }
                }
            }
        }
    }

    
    return;
}