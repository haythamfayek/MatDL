all: col2im im2col

col2im: convnet/col2im_mex.c
		# The following assumes matlab is defined in your path, e.g. /Applications/MATLAB_R2016b.app/bin/matlab
		matlab -nodesktop -nojvm -nosplash -r "cd convnet; mex col2im_mex.c; cd ..; quit"

im2col: convnet/im2col_mex.c
		# The following assumes matlab is defined in your path, e.g. /Applications/MATLAB_R2016b.app/bin/matlab
		matlab -nodesktop -nojvm -nosplash -r "cd convnet; mex im2col_mex.c; cd ..; quit"
