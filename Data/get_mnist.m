% Download and prepare MNIST

%% Download
files = {'train-images-idx3-ubyte', 'train-labels-idx1-ubyte', ...
         't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte'};
for i = 1:numel(files)
   if ~exist(files{i}, 'file')
       url = ['http://yann.lecun.com/exdb/mnist/', files{i}, '.gz'];
       disp(['Downloading ', url]);
       gunzip(url);
       assert(exist(files{i}, 'file') == 2, ['Could not download ', files{i}]);
   end
end

%% Read Images
files = {'train-images-idx3-ubyte', 't10k-images-idx3-ubyte'};
for i = 1:numel(files)
    f = fopen(files{i}, 'rb');
    assert(f ~= -1, ['Could not open ', files{i}]);
    
    magic = fread(f, 1, 'int32', 0, 'ieee-be');
    assert(magic == 2051, ['Bad magic number in ', files{i}]);
    
    numImages = fread(f, 1, 'int32', 0, 'ieee-be');
    numRows = fread(f, 1, 'int32', 0, 'ieee-be');
    numCols = fread(f, 1, 'int32', 0, 'ieee-be');
    
    X{i} = fread(f, inf, 'unsigned char');
    X{i} = reshape(X{i}, numCols, numRows, numImages);
    X{i} = permute(X{i},[3 2 1]);
    X{i} = reshape(X{i}, size(X{i}, 1), size(X{i}, 2) * size(X{i}, 3));
    
    fclose(f); 
end

%% Read Labels
files = {'train-labels-idx1-ubyte', 't10k-labels-idx1-ubyte'};
for i = 1:numel(files)
    f = fopen(files{i}, 'rb');
    assert(f ~= -1, ['Could not open ', files{i}]);

    magic = fread(f, 1, 'int32', 0, 'ieee-be');
    assert(magic == 2049, ['Bad magic number in ', files{i}]);

    numLabels = fread(f, 1, 'int32', 0, 'ieee-be');

    Y{i} = fread(f, inf, 'unsigned char');
    assert(size(Y{i},1) == numLabels, 'Mismatch in label count');

    fclose(f);
end

%% Prepare & Save
train_x = uint8(X{1});
train_y = uint8(dummyvar(Y{1} + 1));
test_x = uint8(X{2});
test_y = uint8(dummyvar(Y{2} + 1));
assert(size(train_x, 1) == size(train_y, 1), 'Mismatch in training set');
assert(size(test_x, 1) == size(test_y, 1), 'Mismatch in test set');
save -v7.3 mnist_uint8 train_x train_y test_x test_y;
