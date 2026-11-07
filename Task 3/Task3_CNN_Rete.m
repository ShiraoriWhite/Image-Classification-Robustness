close all
clear
clc

%% funzione di filtraggio

function imds = filterMacOSFiles(imds)
    % Filtra i file nascosti di macOS e cartelle indesiderate
    % Input: imageDatastore
    % Output: imageDatastore filtrato
    
    validFiles = true(numel(imds.Files), 1);
    
    for i = 1:numel(imds.Files)
        filePath = imds.Files{i};
        % Rimuovi file che contengono __MACOSX o iniziano con ._
        if contains(filePath, '__MACOSX') || contains(filePath, '._')
            validFiles(i) = false;
        end
    end
    
    % Mantieni solo i file validi
    imds = subset(imds, validFiles);
end

%% Funzione di preprocessing (senza augmentation)

function Iout = readAndPreprocessImage(filename, inputSize)
    I = imread(filename);
    
    % Converti in RGB se è grayscale
    if ismatrix(I)
        I = cat(3, I, I, I);
    end
    
    % Resize alle dimensioni di input della rete
    I = imresize(I, inputSize(1:2));
    Iout = I;   % uint8
end

%% Funzione con Data Augmentation (per training)

function Iout = readAndAugmentImage(filename, inputSize)

    % Leggi immagine
    I = imread(filename);

    % Converti in RGB se grayscale
    if ismatrix(I)
        I = cat(3, I, I, I);
    end

    % Resize iniziale
    I = imresize(I, inputSize(1:2));

    % Conversione a double per operazioni
    I = im2double(I);

    % % Applica UNA degradazione principale + degradazioni fotometriche opzionali
    augChoice = randi(4);  % Scegli tra 4 opzioni
    
    switch augChoice
        case 1  % JPEG Compression 
            quality = randi([40, 70]);
            tmpFile = [tempname '.jpg'];
            imwrite(I, tmpFile, 'jpg', 'Quality', quality);
            I = imread(tmpFile);
            delete(tmpFile);
            I = im2double(I);
            
        case 2  % Gaussian Blur
            sigma = 0.5 + rand()*0.8;
            I = imgaussfilt(I, sigma);
            
        case 3  % Gaussian Noise
            noiseLevel = 0.002 + rand()*0.005;
            I = imnoise(I, 'gaussian', 0, noiseLevel);
            
        case 4  % Brightness
            brightness = 0.8 + rand()*0.4;  % 0.8 a 1.2
            I = I * brightness;
    end

    % 1) Vignettatura artificiale (40%)
    if rand < 0.4
        [h, w, ~] = size(I);
        sigma = min(h,w) / 2;
        mask = fspecial('gaussian',[h w], sigma);
        mask = mask / max(mask(:));
        I = I .* mask;
    end

    % 2) Desaturazione colore (50%)
    if rand < 0.5
        Ig = rgb2gray(I);
        I = cat(3, Ig, Ig, Ig);
    end
    
    % Flip orizzontale (geometrico, sempre sicuro)
    if rand() > 0.5
        I = fliplr(I);
    end

    % Clip valori
    I = min(max(I, 0), 1);

    Iout = im2uint8(I);
end

%% Definizione CNN personalizzata (NON pre-addestrata)

fprintf('=== Creazione CNN personalizzata ===\n');

inputSize = [128 128 3];

layers = [
    imageInputLayer(inputSize, ...
    'Name', 'input', ...
    'Normalization', 'zscore')
    
    % Blocco 1
    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')
    
    % Blocco 2
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')
    
    % Blocco 3
    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv3')
    batchNormalizationLayer('Name', 'bn3')
    reluLayer('Name', 'relu3')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool3')
    
    % Blocco 4
    convolution2dLayer(3, 256, 'Padding', 'same', 'Name', 'conv4')
    batchNormalizationLayer('Name', 'bn4')
    reluLayer('Name', 'relu4')
    
    % Classificatore
    globalAveragePooling2dLayer('Name', 'gap')
    dropoutLayer(0.3, 'Name', 'dropout')
    fullyConnectedLayer(100, 'Name', 'fc100')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

lgraph = layerGraph(layers);

fprintf('CNN creata con successo!\n');
fprintf('CNN creata: 4 blocchi conv (32,64,128,256)\n');

% Visualizza l'architettura
 analyzeNetwork(lgraph)

%% Training Set preso dal validation set

validationFolder = '../Dataset/valid';

imds = imageDatastore(validationFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

imds = filterMacOSFiles(imds);

numImages = numel(imds.Files);
numClasses = numel(categories(imds.Labels));

fprintf('Numero di immagini: %d\n', numImages);
fprintf('Numero di classi: %d\n', numClasses);

countEachLabel(imds)

%% Training options
options = trainingOptions('adam', ...
    'MiniBatchSize', 32, ... 
    'MaxEpochs', 150, ...  
    'InitialLearnRate', 5e-3, ...  
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...  
    'LearnRateDropPeriod', 25, ...  
    'L2Regularization', 5e-5, ... 
    'Shuffle', 'every-epoch', ...
    'ValidationPatience', 20, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'parallel');  

%% Training con Data Augmentation

imdsTrain = imds;
imdsTrain.ReadFcn = @(filename)readAndAugmentImage(filename, inputSize);


fprintf('\n=== Inizio training CNN personalizzata con Data Augmentation ===\n');
fprintf('ATTENZIONE: Training da zero richiede molto tempo\n');

tic;
netTransfer = trainNetwork(imdsTrain, lgraph, options);
trainingTime = toc;

fprintf('Training completato in %.1f minuti\n', trainingTime/60);

%% Salva la rete addestrata

% save('cnn_custom_piccola_task3_augmented.mat', 'netTransfer');
% fprintf('Rete salvata in: cnn_custom_piccola_task3_augmented.mat\n');

%% Carica la rete addestrata

%fprintf('=== Caricamento rete addestrata ===\n');

inputSize = [128 128 3];
data = load('Task3_CNN_Pesi.mat');
netTransfer = data.netTransfer;

disp(netTransfer)

%% Test sul dataset degradato

testDegradedFolder = '../Dataset/test_degradato';

imdsTestDegraded = imageDatastore(testDegradedFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames', ...
    'FileExtensions', {'.jpg', '.jpeg', '.png', '.bmp'});

imdsTestDegraded = filterMacOSFiles(imdsTestDegraded);

% Preprocessing per il test (SENZA augmentation)
imdsTestDegraded.ReadFcn = @(filename)readAndPreprocessImage(filename, inputSize);

fprintf('Numero di immagini test degradate: %d\n', numel(imdsTestDegraded.Files));

%% Predizione sul test degradato

[YPred, scores] = classify(netTransfer, imdsTestDegraded);
YTest = imdsTestDegraded.Labels;

%% Valutazione delle performance

% Calcolo Top-1 Accuracy
accuracy = sum(YPred == YTest) / numel(YTest);
fprintf('\n=== RISULTATI FINALI ===\n');
fprintf('Accuracy CNN personalizzata: %.2f%%\n', accuracy * 100);

% Calcolo Top-5 Accuracy
classes = categories(YTest);
numSamples = size(scores,1);

top5Correct = 0;

for i = 1:numSamples
    
    % Ordina le probabilità in ordine decrescente
    [~, idxSorted] = sort(scores(i,:), 'descend');
    
    % Prendi le prime 5 classi predette
    top5Classes = classes(idxSorted(1:5));
    
    % Controlla se la classe vera è tra le prime 5
    if any(top5Classes == YTest(i))
        top5Correct = top5Correct + 1;
    end
end

top5Accuracy = top5Correct / numSamples;

fprintf('Top-5 Accuracy: %.2f%%\n', top5Accuracy * 100);

%% Metriche per classe

classes = categories(YTest);
numClasses = numel(classes);

precision = zeros(numClasses, 1);
recall = zeros(numClasses, 1);
f1score = zeros(numClasses, 1);

for i = 1:numClasses
    TP = sum((YPred == classes{i}) & (YTest == classes{i}));
    FP = sum((YPred == classes{i}) & (YTest ~= classes{i}));
    FN = sum((YPred ~= classes{i}) & (YTest == classes{i}));
    
    precision(i) = TP / (TP + FP + eps);
    recall(i) = TP / (TP + FN + eps);
    f1score(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i) + eps);
end

avgPrecision = mean(precision);
avgRecall = mean(recall);
avgF1Score = mean(f1score);

fprintf('\nMetriche medie:\n');
fprintf('Precision: %.2f%%\n', avgPrecision * 100);
fprintf('Recall: %.2f%%\n', avgRecall * 100);
fprintf('F1-Score: %.2f%%\n', avgF1Score * 100);



%% Visualizza alcuni esempi di predizioni

numSamplesToShow = 9;
idxSamples = randperm(numel(imdsTestDegraded.Files), numSamplesToShow);

figure;
for i = 1:numSamplesToShow
    subplot(3, 3, i);
    
    I = readimage(imdsTestDegraded, idxSamples(i));
    imshow(I);
    
    trueLabel = char(YTest(idxSamples(i)));
    predLabel = char(YPred(idxSamples(i)));
    
    if strcmp(trueLabel, predLabel)
        titleColor = 'g';
    else
        titleColor = 'r';
    end
    
    title(sprintf('True: %s\nPred: %s', trueLabel, predLabel), 'Color', titleColor, 'Interpreter', 'none');
end
sgtitle('Esempi di Predizioni - CNN Personalizzata');
