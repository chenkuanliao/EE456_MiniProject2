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

testImages = testingData';
testLabels = testingLabel;
testImages = reshape(testImages, [32, 32, 3, size(testImages, 2)]);

%------------------------------------------------------------------------
% plot the confusion matrix
load('trained_cnn.mat', 'net2');

confusionMatrix = zeros(10, 10);

for i = 1:12000
    % Make predictions on new data
    image = testImages(:, :, :, i);
    label = testingLabel(i, 1);
    prediction = grp2idx(classify(net2, image));
    % disp(label);
    % disp(prediction);

    confusionMatrix(label+1, prediction) = confusionMatrix(label+1, prediction) + 1;
end

disp("Confusion Matrix");
disp(confusionMatrix);