%% Simple Face Recognition Example
%  Copyright 2014-2015 The MathWorks, Inc.
%% Load Image Information from ATT Face Database Directory
faceDatabase = imageSet('FaceDatabaseATT','recursive');

%% Display Montage of First Face
figure;
montage(faceDatabase(4).ImageLocation)
title('Images of Single Face');
%%  Display Query Image and Database Side-Side
personToQuery = 4;
galleryImage = read(faceDatabase(personToQuery),2);
figure;
imageList = {faceDatabase(1).ImageLocation{5}};
for i=2:size(faceDatabase,2)
    imageList{i,:} =  faceDatabase(i).ImageLocation{8};
end
subplot(1,2,1);imshow(galleryImage);
subplot(1,2,2);montage(imageList);
diff = zeros(1,9);
%% Split Database into Training & Test Sets
[training,test] = partition(faceDatabase,[0.8 0.2]);
 
%% Extract and display Histogram of Oriented Gradient Features for single face 
person = 10;
[hogFeature,visualization]= ...
extractHOGFeatures(read(training(person),1));
figure;
subplot(2,1,1);imshow(read(training(person),1));title('Input Face');
subplot(2,1,2);plot(visualization);title('HoG Feature');
%% Extract HOG Features for training set 
trainingFeatures = zeros(size(training,2)*training(1).Count,4680);
featureCount = 1;
trainingLabel = {};
for i=1:size(training,2)
     for j = 1:training(i).Count
         trainingFeatures(featureCount,:) = extractHOGFeatures(read(training(i),j));
         trainingLabel{featureCount} = training(i).Description;    
         featureCount = featureCount + 1;
     end
     personIndex{i} = training(i).Description;
end
 whos trainingFeatures
 whos trainingLabel
%% Create 40 class classifier using fitcecoc 

faceClassifier = fitcecoc(trainingFeatures,trainingLabel);


%% Test Images from Test Set 
person = 10;
queryImage = read(test(person),2);
queryFeatures = extractHOGFeatures(queryImage);
personLabel = predict(faceClassifier,queryFeatures);
% Map back to training set to find identity 
booleanIndex = strcmp(personLabel, personIndex);
integerIndex = find(booleanIndex);
subplot(1,2,1);imshow(queryImage);title('Query Face');
subplot(1,2,2);imshow(read(training(integerIndex),1));title('Matched Class');

%% Test First 5 People from Test Set
figureNum = 1;
for person=1:5
    figure;
    for j = 1:test(person).Count
        queryImage = read(test(person),j);
        queryFeatures = extractHOGFeatures(queryImage);
        personLabel = predict(faceClassifier,queryFeatures);
        % Map back to training set to find identity
        booleanIndex = strcmp(personLabel, personIndex);
        integerIndex = find(booleanIndex);
        subplot(2,2,figureNum);imshow(imresize(queryImage,3));title('Query Face');
        subplot(2,2,figureNum+1);imshow(imresize(read(training(integerIndex),1),3));title('Matched Class');
        figureNum = figureNum+2;
        
    end
    figureNum = 1;

end

