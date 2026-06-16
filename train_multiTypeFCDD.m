%% 1. Configuration & Environment Setup
clc; clear all; close all;

% Execution Flags & Hyperparameters
trainFlag = 1;
upsampleMethodFlag = false; % Set true if using receptive_upsample

% Paths & Dataset Settings
dataDir = fullfile(pwd, "WRc_Dataset");
anomalyLabels = ["Connection", "Deposit", "Displaced Joint", "Fracture", "Roots"];
classOrder = anomalyLabels;

% Load Backbone & Image Size
backbone = pretrainedEncoderNetwork("inceptionv3", 3);
imgSize = backbone.Layers(1).InputSize(1:2);

%% 2. Data Ingestion & Splitting
imds = imageDatastore(dataDir, IncludeSubfolders=true, LabelSource="foldernames");

% Filter for relevant labels (Normal + Target Anomalies)
mask = ismember(imds.Labels, ["Normal", anomalyLabels]);
imds_subset = subset(imds, find(mask));
imds_subset.Labels = removecats(imds_subset.Labels);

% Partition Dataset (70/10/20 Split)
rng(0);
trainRatio = 0.7; calRatio = 0.1;
[imdsTrain, imdsCal, imdsTest] = splitEachLabel(imds_subset, trainRatio, calRatio, "randomized", Include=anomalyLabels);

% Configure Calibration & Test Datastores
imdsCal.ReadFcn  = @(x) resizeImage(imread(x), imgSize);
imdsTest.ReadFcn = @(x) resizeImage(imread(x), imgSize);
dsCal = transform(imdsCal, @addLabelData, IncludeInfo=true);

%% 3. Data Augmentation (Training Set)
classNames = categories(imdsTrain.Labels);
numAnomalyClasses = numel(classNames);

% Balance the multi-class training data
classDSList = cell(1, numAnomalyClasses);
for i = 1:numAnomalyClasses
    classDSList{i} = shuffle(subset(imdsTrain, imdsTrain.Labels == classNames{i}));
end
dsTrain = balancedMultiClassDatastore(classDSList, classNames);

% Define Augmentation Parameters
augmentParams = struct(...
    'AugProb', 0.5, 'RandRotation', [-15, 15], ...
    'RandXTranslation', [-20, 20], 'RandYTranslation', [-20, 20], ...
    'RandXReflection', true, 'RandYReflection', false, ...
    'RandBrightness', [0.8, 1.2], 'RandContrast', true, ...
    'RandColorJitter', false, 'RandGaussianBlur', [], 'RandNoise', false);

augmentFcn = @(data, info) augmentImageData(data, info, imgSize, augmentParams);
imdsTrainAugmented = transform(dsTrain, augmentFcn, IncludeInfo=true);

% Optional Visualization (Montage)
imdsTrainAugmented_sample = transform(imdsTrain, augmentFcn, IncludeInfo=true);
exampleData = readall(subset(imdsTrainAugmented_sample, 696:704));
figure; montage(exampleData(:,1)); title("Augmented Training Samples");

%% 4. Construct FCDD Network Architecture
backbone = freezeLayers(backbone);

fcddHead = [
    convolution2dLayer(3, 512, Padding="same", Name='fcddHeadConv1', WeightsInitializer='he')
    batchNormalizationLayer(Name='fcddHeadBN1')
    reluLayer(Name='fcddHeadRelu1')
    convolution2dLayer(3, 512, Padding="same", Name='fcddHeadConv2', WeightsInitializer='he')
    batchNormalizationLayer(Name='fcddHeadBN2')
    reluLayer(Name='fcddHeadRelu2')   
    convolution2dLayer(1, numAnomalyClasses, Name='fcddHeadFinal1x1Conv', WeightsInitializer='he')
    functionLayer(@customAnomalyScore, Name="anomalyScoreLayer")
];

fcddNet = addLayers(backbone, fcddHead);
fcddNet = connectLayers(fcddNet, "mixed2", "fcddHeadConv1");

%% 5. Model Training
mbqTrain = minibatchqueue(imdsTrainAugmented, 2, MiniBatchSize=64, ...
    MiniBatchFcn=@(x,t) preprocessMiniBatch(x, t, classOrder), MiniBatchFormat=["SSCB", ""]);
mbqVal = minibatchqueue(dsCal, 2, MiniBatchSize=64, ...
    MiniBatchFcn=@(x,t) preprocessMiniBatch(x, t, classOrder), MiniBatchFormat=["SSCB", ""]);

options = trainingOptions("adam", Shuffle="every-epoch", MaxEpochs=1000, ...
    InitialLearnRate=1e-4, LearnRateDropFactor=0.1, LearnRateDropPeriod=500, ...
    LearnRateSchedule="piecewise", MiniBatchSize=64, BatchNormalizationStatistics="moving", ...
    ValidationData=mbqVal, ResetInputNormalization=false, Plots="training-progress", Verbose=true);

if trainFlag
    detector = trainnet(mbqTrain, fcddNet, @(Y,T) multiTypeFcddLoss(Y,T), options);
else
    load trainedMultiTypeFCDDNet_augmentedWRcv1_inceptionv3.mat detector;
end

%% 6. Calibration & Optimal Threshold Determination
numCalFiles = length(imdsCal.Files);
scoresCal = zeros(numAnomalyClasses, numCalFiles);

for i = 1:numCalFiles
    A = predict(detector, single(readimage(imdsCal, i)));
    scoresCal(:, i) = squeeze(mean(A, [1, 2]));
end

% Logical matrix conversion for Calibration targets
targetLabels = false(numAnomalyClasses, numel(imdsCal.Labels));

for c = 1:numAnomalyClasses
    targetLabels(c,:) = ismember(imdsCal.Labels, categorical(anomalyLabels(c)));
end

% Compute Thresholds & Metrics
thresholds = zeros(numAnomalyClasses, 1);
figure('Name', 'Calibration Histograms'); tiledlayout('flow');
figure('Name', 'ROC Curves'); tiledlayout('flow');

for c = 1:numAnomalyClasses
    % Determine Thresholds via Max F1 Score
    thresholds(c) = anomalyThreshold(targetLabels(c,:), scoresCal(c,:), true, "MaxF1Score");
    
    % Plot Histogram
    figure(2); nexttile;
    [~, edges] = histcounts(scoresCal, 20);
    histogram(scoresCal(c, ~targetLabels(c,:)), edges, FaceAlpha=0.5, DisplayName='Normal'); hold on;
    histogram(scoresCal(c, targetLabels(c,:)), edges, FaceAlpha=0.5, DisplayName=anomalyLabels(c));
    title("Class: " + anomalyLabels(c)); xlabel("Mean Score"); ylabel("Counts"); legend; hold off;
    
    % Plot ROC Metrics
    figure(3); nexttile;
    roc = rocmetrics(targetLabels(c, :), scoresCal(c, :), true);
    plot(roc); title(sprintf("ROC AUC (%s): %.3f", anomalyLabels(c), roc.AUC));
end

%% 7. Test Set Evaluation
numTestFiles = length(imdsTest.Files);
scoresTest = zeros(numAnomalyClasses, numTestFiles);

for i = 1:numTestFiles
    A = predict(detector, single(readimage(imdsTest, i)));
    scoresTest(:, i) = squeeze(mean(A, [1, 2]));
end

testSetOutputLabels = scoresTest >= thresholds;

for c = 1:numAnomalyClasses
    metrics = evaluateAnomalyDetection(testSetOutputLabels(c,:), imdsTest.Labels, anomalyLabels(c));
    M = metrics.ConfusionMatrix{:,:};
    acc = sum(diag(M)) / sum(M,"all");
    fprintf("[%s] Evaluation Accuracy: %.4f\n", anomalyLabels(c), acc);
end

%% 8. Explainability & Spatial Anomaly Heatmaps
% Establish global display min/max normalization mappings dynamically
minMapVals = inf(numAnomalyClasses, 1);
maxMapVals = -inf(numAnomalyClasses, 1);
reset(imdsCal);

while hasdata(imdsCal)
    img = single(read(imdsCal));
    mapLowRes = predict(detector, img);
    for c = 1:numAnomalyClasses
        map = imresize(mapLowRes(:,:,c), size(img,1:2), "bilinear");
        minMapVals(c) = min(min(map, [], "all"), minMapVals(c));
        maxMapVals(c) = max(max(map, [], "all"), maxMapVals(c));
    end
end

targetLabels = false(numAnomalyClasses, numel(imdsTest.Labels));

for c = 1:numAnomalyClasses
    testSetAnomalyLabels(c,:) = ismember(imdsTest.Labels, categorical(anomalyLabels(c)));
end

for c = 1:numAnomalyClasses
    idxTruePositive = find(testSetAnomalyLabels(c,:) & testSetOutputLabels(c,:));
    if isempty(idxTruePositive)
        warning("No true positive matches found for class: %s", anomalyLabels(c));
        continue;
    end
    
    % Grab an instance index safely (defaulting to the 9th instance or the last available)
    targetIdx = idxTruePositive(min(9, length(idxTruePositive)));
    img = read(subset(imdsTest, targetIdx));
    mapLowRes = predict(detector, single(img));
    
    figure('Name', "Heatmap Analysis: " + anomalyLabels(c));
    tiledlayout(1, numAnomalyClasses + 1, TileSpacing="tight", Padding="tight");
    
    nexttile; imshow(img); title(anomalyLabels(c));
    
    for k = 1:numAnomalyClasses
        map = imresize(mapLowRes(:,:,k), size(img,1:2), "bilinear");
        nexttile;
        imshow(anomalyMapOverlay(img, map, MapRange=[minMapVals(k), maxMapVals(k)], Blend="equal"));
        title(anomalyLabels(k) + " Heatmap");
    end
end
%% 9. Save Model Parameters

save('trainedMultiTypeFCDDNet.mat', 'detector','minMapVals','maxMapVals','thresholds');

%% 10. Helper Functions
function [data, info] = addLabelData(data, info)
    data = {data, info.Label};
end

function net = freezeLayers(net)
    learnables = net.Learnables;
    for i = 1:size(learnables, 1)
        layerName = learnables.Layer(i);
        paramName = learnables.Parameter(i);
        if ~ismember(paramName, ["Scale", "Offset"])
            net = setLearnRateFactor(net, layerName, paramName, 0);
        end
    end
end

function [X, T] = preprocessMiniBatch(dataX, dataT, classOrder)
    X = cat(4, dataX{:});
    labels = categorical(vertcat(dataT{:}), classOrder);
    T = permute(onehotencode(labels, 2), [3 4 2 1]); % Convert directly to SSCB format [1 x 1 x C x B]
end

function loss = multiTypeFcddLoss(Y, T)
    numAnomalyClasses = size(T, 3);
    loss = 0;
    normalTerm = mean(Y, [1 2]);
    for i = 1:numAnomalyClasses
        anomalyTerm = log(1 - exp(-normalTerm(:,:,i,:)) + eps('single'));
        isGood = ~T(:,:,i,:);
        loss = loss + mean(single(isGood) .* normalTerm(:,:,i,:) - single(~isGood) .* anomalyTerm, 'all');
    end
end