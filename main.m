clear;
clc;
close all;
% Deep learning ToolBox is required 

%----------------------------------------------
% handle the data
load data\batches.meta.mat;
labelNames = label_names; % 10x1

AllData = zeros(60000, 3072);
AllLabel = zeros(60000, 1);

load data/data_batch_1.mat;
AllData(1:10000, :) = data;
AllLabel(1:10000, :) = labels;

load data/data_batch_2.mat;
AllData(10001:20000, :) = data;
AllLabel(10001:20000, :) = labels;

load data/data_batch_3.mat;
AllData(20001:30000, :) = data;
AllLabel(20001:30000, :) = labels;

load data/data_batch_4.mat;
AllData(30001:40000, :) = data;
AllLabel(30001:40000, :) = labels;

load data/data_batch_5.mat;
AllData(40001:50000, :) = data;
AllLabel(40001:50000, :) = labels;

load data/test_batch.mat
AllData(50001:60000, :) = data;
AllLabel(50001:60000, :) = labels;

% we will split the data into 4:1 = trainind:testing
% within the training data, will will split the data into 4:1 =
% training:validation 
rawTrainingData = AllData(1:38400, :);
rawValidationData = AllData(38401:48000, :);
rawTestingData = AllData(48001:60000, :);

trainingLabel = AllLabel(1:38400, :);
validationLabel = AllLabel(38401:48000, :);
testingLabel = AllLabel(48001:60000, :);

% change the vectors to image dimentions (32x32x3)
trainingData = zeros(32, 32, 3, 38400);
validationData = zeros(32, 32, 3, 9600);
testingData = zeros(32, 32, 3, 12000);

len = size(rawTrainingData);
for i = 1:len(1)
    vector = rawTrainingData(i, :);
    matrix = vectorToImage(vector);
    trainingData(:, :, :, i) = matrix;
end

len = size(rawValidationData);
for i = 1:len(1)
    vector = rawValidationData(i, :);
    matrix = vectorToImage(vector);
    validationData(:, :, :, i) = matrix;
end

len = size(rawTestingData);
for i = 1:len(1)
    vector = rawTestingData(i, :);
    matrix = vectorToImage(vector);
    testingData(:, :, :, i) = matrix;
end

% % Trying to display the image
% imageMatrix = reshape(image, [], 3);
% disp(size(imageMatrix));
% imageRGB = reshape(imageMatrix, [32, 32, 3]);
% imshow(imageRGB);


%----------------------------------------------
% training the CNN
XTrain = AllData(1:1000);
YTrain = AllLabel(1:1000);

XValidation = AllData(1001:1100);
YValidation = AllLabel(1001:1100);

% Define the CNN architecture
layers = [
    % input layer
    imageInputLayer([1 3072])

    % Layer 1
    convolution2dLayer(7, 10)     % Convolutional layer with 10 filters of zize 7x7
    reluLayer()                                      % ReLU activation layer
    maxPooling2dLayer(2)                             % Max pooling layer

    % Layer 2
    convolution2dLayer(7, 10)     % Convolutional layer with 10 filters of zize 7x7
    reluLayer()                                      % ReLU activation layer
    maxPooling2dLayer(2)                             % Max pooling layer

    % Layer 3
    convolution2dLayer(5, 10)     % Convolutional layer with 10 filters of zize 5x5
    reluLayer()                                      % ReLU activation layer
    maxPooling2dLayer(2)                             % Max pooling layer

    % Layer 4
    convolution2dLayer(5, 10)     % Convolutional layer with 10 filters of zize 5x5
    reluLayer()                                      % ReLU activation layer
    maxPooling2dLayer(2)                             % Max pooling layer

    % Layer 5
    convolution2dLayer(3, 10)     % Convolutional layer with 10 filters of zize 3x3
    reluLayer()                                      % ReLU activation layer
    maxPooling2dLayer(2)                             % Max pooling layer

    % Layer 6
    convolution2dLayer(3, 10)     % Convolutional layer with 10 filters of zize 3x3
    reluLayer()                                      % ReLU activation layer
    maxPooling2dLayer(2)                             % Max pooling layer

    fullyConnectedLayer(10)                          % Output layer with sigmoid activation for binary classification
    classificationLayer()
];

% Define options for training the network
options = trainingOptions('sgdm', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 64, ...
    'InitialLearnRate', 0.001, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

net = trainNetwork(XTrain, YTrain, layers, options);

%----------------------------------------------
% testing the CNN


%----------------------------------------------
% plott the graphs

