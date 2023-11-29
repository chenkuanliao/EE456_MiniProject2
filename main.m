clear;
clc;
close all;

%----------------------------------------------
% handle the data
load data\batches.meta.mat
labelNames = label_names;
disp(labelNames); % 10x1

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

validationTable = table(validationData, categorical(validationLabel), 'VariableNames', {'Features', 'Label'});
trainingData = reshape(trainingData', [32, 32, 3, numel(trainingLabel)]);

%----------------------------------------------
% training the CNN
inputSize = [32 32 3];
numClasses = 10;

layers = [
    imageInputLayer(inputSize)
    convolution2dLayer(5,20)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm', ...
    'MaxEpochs',4, ...
    'ValidationData',validationTable , ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');
disp(size(trainingData));
disp(size(trainingLabel));

net = trainNetwork(trainingData, categorical(trainingLabel), layers, options);

%----------------------------------------------
% testing the CNN
YPred = classify(net, testingData);
YValidation = categorical(testingLabel);  % Convert testing labels to categorical
accuracy = mean(YPred == YValidation);
