close all
clear
clc

%% Funzione di filtraggio

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
    Iout = I;
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

    % % Applica UNA degradazione principale
    augChoice = randi(6);  % Scegli tra 6 opzioni
    
    switch augChoice
        case 1  % JPEG Compression 
            quality = randi([60, 85]);
            tmpFile = [tempname '.jpg'];
            imwrite(I, tmpFile, 'jpg', 'Quality', quality);
            I = imread(tmpFile);
            delete(tmpFile);
            I = im2double(I);
            
        case 2  % Gaussian Blur
            sigma = 0.3 + rand()*0.5;
            I = imgaussfilt(I, sigma);
            
        case 3  % Gaussian Noise
            noiseLevel = 0.001 + rand()*0.003;
            I = imnoise(I, 'gaussian', 0, noiseLevel);
            
        case 4  % Brightness
            brightness = 0.8 + rand()*0.4;  % 0.8 a 1.2
            I = I * brightness;

        case 5 % Vignettatura artificiale
            [h, w, ~] = size(I);
            sigma = min(h,w) / 2;
            mask = fspecial('gaussian',[h w], sigma);
            mask = mask / max(mask(:));
            I = I .* mask;
        
        case 6 % Desaturazione colore
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

%% Funzione di leggero preprocessing

function Iout = readAndEnhanceTestImage(filename, inputSize)
    I = imread(filename);

    if ismatrix(I)
        I = cat(3, I, I, I);
    end

    I = imresize(I, inputSize(1:2));
    I = im2double(I);

    % --- Correzione vignettatura (molto leggera)
    [h,w,~] = size(I);
    sigma = min(h,w);
    mask = fspecial('gaussian',[h w], sigma);
    mask = mask / max(mask(:));
    I = I ./ (0.6 + 0.4*mask);   % evita overcorrection

    % --- Attenuazione desaturazione (senza oversaturation)
    Ig = rgb2gray(I);
    alpha = 0.15;  % quanto reinserire informazione cromatica
    I = (1 - alpha) * I + alpha * cat(3, Ig, Ig, Ig);

    % Clip di sicurezza
    I = min(max(I,0),1);
    Iout = im2uint8(I);
end

%% 1. CONFIGURAZIONE E CARICAMENTO DATI

fprintf("Caricamento dataset...\n");

trainDir = '../Dataset/train';
valDir = '../Dataset/valid';

% Caricamento Datastore
trainSet = imageDatastore(trainDir, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

valSet = imageDatastore(valDir, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Filtra file macOS
trainSet = filterMacOSFiles(trainSet);
valSet = filterMacOSFiles(valSet);

numClasses = numel(categories(trainSet.Labels));

% Definiamo una dimensione standard per le immagini 
% Una dimensione minore velocizza l'addestramento da zero.
inputSize = [64 64 3]; 

%% 2. DEFINIZIONE ARCHITETTURA CNN (Da Zero)

fprintf('=== Creazione CNN personalizzata ===\n');

% Creiamo una struttura a blocchi: Conv -> BatchNormalizaton -> ReLU -> MaxPool
layers = [
    imageInputLayer(inputSize, 'Name', 'input')
    
    % Blocco 1 - Nota: WeightsInitializer 'he' evita che i gradienti spariscano
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv_1')
    batchNormalizationLayer('Name', 'bn_1')
    reluLayer('Name', 'relu_1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool_1')
    
    % Blocco 2
    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv_2')
    batchNormalizationLayer('Name', 'bn_2')
    reluLayer('Name', 'relu_2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool_2')
    
    % Blocco 3
    convolution2dLayer(3, 256, 'Padding', 'same', 'Name', 'conv_3')
    batchNormalizationLayer('Name', 'bn_3')
    reluLayer('Name', 'relu_3')
    dropoutLayer(0.4, 'Name', 'drop_1')
    
    
    fullyConnectedLayer(numClasses, 'Name', 'fc_out')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

fprintf('CNN creata con successo!\n');

% Visualizza l'architettura
analyzeNetwork(layers);

%% 3. OPZIONI DI ADDESTRAMENTO

options = trainingOptions('adam', ...      
    'MaxEpochs', 30, ...                  
    'MiniBatchSize', 64, ...
    'InitialLearnRate', 0.001, ...
    'L2Regularization', 1e-3, ...
    'LearnRateSchedule', 'piecewise', ...  
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 10, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', valSet, ...
    'ValidationFrequency', 50, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ValidationPatience', 8, ...           
    'OutputNetwork', 'best-validation-loss', ... 
    'ExecutionEnvironment', 'parallel');

%% 4. ESECUZIONE TRAINING

fprintf("Inizio addestramento da zero (Task 2)...\n");

trainSet.ReadFcn = @(filename) readAndAugmentImage(filename, inputSize);
valSet.ReadFcn   = @(filename) readAndPreprocessImage(filename, inputSize);

tic;
trainedNet = trainNetwork(trainSet, layers, options);

trainingTime = toc;

fprintf('Training completato in %.1f minuti\n', trainingTime/60);

%% Salva la rete addestrata

%save('Task2_CNN_Pesi.mat', 'trainedNet');
%fprintf('Rete salvata in: Task2_CNN_Pesi.mat\n');

%% Carica la rete addestrata

% fprintf('=== Caricamento rete addestrata ===\n');
% 
% inputSize = [64 64 3];
% data = load('Task2_CNN_Pesi.mat');
% trainedNet = data.netTransfer;
% 
% disp(trainedNet)

%% 5. VALUTAZIONE SUL TEST SET

fprintf("Valutazione finale...\n");

testDir  = '../Dataset/test_degradato';

testSet = imageDatastore(testDir, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

testSet = filterMacOSFiles(testSet);
testSet.ReadFcn  = @(filename) readAndEnhanceTestImage(filename, inputSize);

[YPred, scores] = classify(trainedNet, testSet);
YTest = testSet.Labels;

%% Valutazione delle performance

accuracy = mean(YPred == YTest);
fprintf('Accuratezza Task 2: %.2f%%\n', accuracy * 100);

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

%% Matrice di confusione

figure;
confusionchart(YTest, YPred, ...
    'Title', 'Confusion Matrix - Trained from Scratch');

%% Visualizza alcuni esempi di predizioni

numSamplesToShow = 9;
idxSamples = randperm(numel(testSet.Files), numSamplesToShow);

figure;
for i = 1:numSamplesToShow
    subplot(3, 3, i);
    
    I = readimage(testSet, idxSamples(i));
    imshow(I);
    
    trueLabel = char(YTest(idxSamples(i)));
    predLabel = char(YPred(idxSamples(i)));
    
    if strcmp(trueLabel, predLabel)
        titleColor = 'g';
    else
        titleColor = 'r';
    end
    
    title(sprintf('True: %s\nPred: %s', trueLabel, predLabel), ...
        'Color', titleColor, ...
        'Interpreter', 'none');
end
sgtitle('Esempi di Predizioni - CNN Personalizzata');
