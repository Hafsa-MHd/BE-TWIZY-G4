clear all; 
close all; 
clc;


pathTrainDataset = "O:\BE_Twizy\BE_Matlab\train_dataset"; 
sourceDir        = "O:\BE_Twizy\BE_Matlab\test";
destBad          = "O:\BE_Twizy\BE_Matlab\PasValid";
destGood         = "O:\BE_Twizy\BE_Matlab\valid";
baseDestSpeed    = "O:\BE_Twizy\BDD_Roboflow_Tazi\TestSpeed"; 



seuilGmag = 10.0;
seuilLap  = 50.0;
lumiMin   = 45;
lumiMax   = 220;


filesTrain = [dir(fullfile(pathTrainDataset,'*.jpg'))];

X_train_list = {};
Y_train_list = strings(0,1);


for f = 1:length(filesTrain)
    imgName  = filesTrain(f).name;
    fullPath = fullfile(pathTrainDataset, imgName);
    try
        I = imread(fullPath);
        if size(I,3) ~= 3
    		continue;
	end


%%HSV
        ihsv = rgb2hsv(I);
        Ih = ihsv(:,:,1);
        masqueR = (Ih > 0.95) | (Ih < 0.03);
        masqueJ = (Ih < 0.18) & (Ih > 0.05);
        masque = masqueR | masqueJ;
        se = strel("disk",2);
	masqueerode  = imerode(masque,se);
        masquedilate = imdilate(masqueerode,se);
        contourim = edge(masquedilate);


        [centers,radii] = imfindcircles(contourim,[8 80],'Sensitivity',0.97);



        if ~isempty(centers)
            [~, idxBiggest] = max(radii);
            center = centers(idxBiggest,:);
            radius = radii(idxBiggest);


            rowMin = max(1, round(center(2)-radius));
            rowMax = min(size(I,1), round(center(2)+radius));
            colMin = max(1, round(center(1)-radius));
            colMax = min(size(I,2), round(center(1)+radius));


            if (rowMax > rowMin) && (colMax > colMin)

                panneauCrop = I(rowMin:rowMax, colMin:colMax,:);
                panneauResized = imresize(panneauCrop,[64 64]);
                prefix = imgName(1:5);
                if all(isstrprop(prefix,'digit'))
                    classID = string(prefix);
                else
                    classID = "00003";
                end
                X_train_list{end+1} = single(panneauResized)/255;
                Y_train_list(end+1,1) = classID;

            end
        end
    end
end


nbImagesTrain = length(X_train_list);
X_train = zeros(64,64,3,nbImagesTrain,'single');
for idx = 1:nbImagesTrain
    X_train(:,:,:,idx) = X_train_list{idx};
end


uniqueClasses = unique(Y_train_list);
nbClasses = length(uniqueClasses);
classesNames = strings(nbClasses,1);

for c = 1:nbClasses
    switch uniqueClasses(c)
        case "00000"
            classesNames(c) = "Panneau20";
        case "00001"
            classesNames(c) = "Panneau30";
        case "00002"
            classesNames(c) = "Panneau50";
        case "00003"
            classesNames(c) = "Panneau60";
        case "00004"
            classesNames(c) = "Panneau70";
        case "00005"
            classesNames(c) = "Panneau80";
        otherwise
            classesNames(c) = "Classe_" + uniqueClasses(c);
    end
end


Y_train = categorical(Y_train_list, uniqueClasses, classesNames);


%%Architecture

layersCNN = [
    imageInputLayer([64 64 3], 'Normalization', 'none', 'Name', 'input')
    
    convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool1')
    
    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool2')
    
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv3')
    reluLayer('Name', 'relu3')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool3')
    
    fullyConnectedLayer(nbClasses, 'Name', 'fc_final')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];


options = trainingOptions('adam','MiniBatchSize',min(32,nbImagesTrain),'MaxEpochs',12,'Verbose',true,'Plots','training-progress');
net = trainNetwork(X_train, Y_train, layersCNN, options);
YPred = classify(net, X_train);

accuracy = mean(YPred == Y_train);

fprintf('Accuracy entraînement : %.2f %%\n', accuracy*100);





fileList = [dir(fullfile(sourceDir,'*.jpg'))];
for i = 1:length(fileList)
    imgName  = fileList(i).name;
    fullPath = fullfile(sourceDir,imgName);
    try
        I = imread(fullPath);
        if size(I,3) ~= 3
    		continue;
	end
        Igray = rgb2gray(I);
%%Laplacian
        lapFilter = [0 1 0;
                     1 -4 1;
                     0 1 0];
        Iapp = imfilter(double(Igray),lapFilter,'replicate');
        scoreFlou = var(Iapp(:));

        [~, Gmag] = imgradient(Igray);
        scoreContraste = mean(Gmag(:));
        scoreLumi = mean(Igray(:));
        if (scoreFlou > seuilLap) && (scoreContraste > seuilGmag) && (scoreLumi > lumiMin) && (scoreLumi < lumiMax)
%%HSV
            ihsv = rgb2hsv(I);
            Ih = ihsv(:,:,1);
            masqueR = (Ih > 0.95) | (Ih < 0.03);
            masqueJ = (Ih < 0.18) & (Ih > 0.05);
            masque = masqueR | masqueJ;
            se = strel("disk",2);
            masque = imerode(masque,se);
            masque = imdilate(masque,se);
            contourim = edge(masque);

            [centers,radii] = imfindcircles(contourim,[8 80],'Sensitivity',0.97);

            if ~isempty(centers)
                [~, idxBiggest] = max(radii);
                center = centers(idxBiggest,:);
                radius = radii(idxBiggest);
                rowMin = max(1, round(center(2)-radius));
                rowMax = min(size(I,1), round(center(2)+radius));
                colMin = max(1, round(center(1)-radius));
                colMax = min(size(I,2), round(center(1)+radius));
                if (rowMax > rowMin) && (colMax > colMin)
                    panneauTest = I(rowMin:rowMax, colMin:colMax,:);
                    Iresized = imresize(panneauTest,[64 64]);
                    Inormalized = single(Iresized)/255;
                    Inormalized = reshape(Inormalized,[64 64 3 1]);

                    [labelPred,~] = classify(net, Inormalized);
                    nomClasse = string(labelPred);


                    dossierCible = fullfile(baseDestSpeed,nomClasse);
                    destinationFile = fullfile( dossierCible,imgName);
                    if ~exist(destinationFile,'file')
                        copyfile(fullPath, destinationFile);
                    end
                end
	    end
	end
    catch ME
    end
end

