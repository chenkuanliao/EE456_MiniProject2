clear;
clc;
close all;

%----------------------------------------------
% handle the data
load data\batches.meta.mat
labelNames = label_names;

AllData = zeros(60000, 3072);
AllLabel = zeros(60000, 1);

load data/data_batch_1.mat
AllData(1:10000, :) = data;
AllLabel(1:10000, :) = labels;

load data/data_batch_2.mat
AllData(10001:20000, :) = data;
AllLabel(10001:20000, :) = labels;

load data/data_batch_3.mat
AllData(20001:30000, :) = data;
AllLabel(20001:30000, :) = labels;

load data/data_batch_4.mat
AllData(30001:40000, :) = data;
AllLabel(30001:40000, :) = labels;

load data/data_batch_5.mat
AllData(40001:50000, :) = data;
AllLabel(40001:50000, :) = labels;

load data/test_batch.mat
AllData(50001:60000, :) = data;
AllLabel(50001:60000, :) = labels;

% we will split the data into 4:1 = trainind:testing
% within the training data, will will split the data into 4:1 =
% training:validation 
trainingData = AllData(1:38400, :);
validationData = AllData(38401:48000, :);
testingData = AllData(48001:60000, :);

trainingLabel = AllLabel(1:38400, :);
validationLabel = AllLabel(38401:48000, :);
testingLabel = AllLabel(48001:60000, :);



%----------------------------------------------
% training the CNN
trainImages = trainingData';
trainLabels = categorical(trainingLabel);

validationImages = validationData';
validationLabels = categorical(validationLabel);

testImages = testingData';
testLabels = categorical(testingLabel);

% Reshape the data to 32x32x3 (height x width x channels)
trainImages = reshape(trainImages, [32, 32, 3, size(trainImages, 2)]);
validationImages = reshape(validationImages, [32, 32, 3, size(validationImages, 2)]);
testImages = reshape(testImages, [32, 32, 3, size(testImages, 2)]);

% Define the CNN architecture
layers = [
    % input layer
    imageInputLayer([32, 32, 3])

    % layer 1
    convolution2dLayer(3, 64, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2, 'Stride', 2)

    % layer 2
    convolution2dLayer(3, 128, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2, 'Stride', 2)

    % layer 3
    convolution2dLayer(3, 256, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2, 'Stride', 2)

    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer
];

% Specify training options
options = trainingOptions('sgdm', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 128, ...
    'ValidationData', {validationImages, validationLabels}, ...
    'ValidationFrequency', 5, ...
    'Plots', 'training-progress', ...
    'Verbose', true);

% Train the CNN
net = trainNetwork(trainImages, trainLabels, layers, options);

% Evaluate on the test set
predictedLabels = classify(net, testImages);
accuracy = sum(predictedLabels == testLabels) / numel(testLabels);
fprintf('Accuracy on the test set: %.2f%%\n', accuracy * 100);

% Save the trained network to a file
save('trained_network.mat', 'net');
