%% Simple Face Recognition using IAET face Database
%  Copyright 2015-2016 mohammed_elganzory@yahoo.com, e-mail.
%% Load Image Information from wadi_degla Face Database Directory
clear ; close all ; clc
faceGallery = imageSet('Data_base','recursive');
% imshow(read(faceGallery(14),1))

galleryNames = {faceGallery.Description};
displayFaceGallery(faceGallery,galleryNames);
trainingFeatures = [];
trainingLabel = {0};
featureCount = 1;
Facedetector = vision.CascadeObjectDetector(); %create a face detector object
% try
%     Loa = load('Database.mat');
%     faceClassifier = Loa.faceClassifier;
%     personIndex = loa.personIndex; 
% catch
size(faceGallery,2)

for i=1:size(faceGallery,2)
faceGallery(i).Count
    for j = 1:faceGallery(i).Count
        A = read(faceGallery(i),j);
        BBox= step(Facedetector, A)
        mergeface1 = []; % to reject any un correct face detected
        if ~isempty(BBox) && size(BBox,1)>1 ;
            for ii=1:size(BBox,1) ; mergeface1 = [mergeface1 max(BBox(ii,:))-min(BBox(ii,:))]; 
            end    
            realface = find(mergeface1 == min(mergeface1))
            figure
            imcr = imcrop(A,BBox(realface,:));imshow(imcr)
%            sizeNormalizedImage = imresize(rgb2gray(imcr),[150 150]);
%            trainingFeatures(featureCount,:) = extractHOGFeatures(sizeNormalizedImage);
%            trainingLabel{featureCount} = faceGallery(i).Description;    
%            featureCount = featureCount + 1;

        elseif isempty(BBox);continue;
        else 
            figure
            imcr = imcrop(A,BBox(1,:));imshow(imcr)
%            sizeNormalizedImage = imresize(rgb2gray(imcr),[150 150]);
%            trainingFeatures(featureCount,:) = extractHOGFeatures(sizeNormalizedImage);
%            trainingLabel{featureCount} = faceGallery(i).Description;    
%            featureCount = featureCount + 1;

        end     
        
        
        
    end
%     personIndex{i} = faceGallery(i).Description;
    
end
%% Create Classifier 
% faceClassifier = fitcecoc(trainingFeatures,trainingLabel);
% save('Database.mat','faceClassifier','personIndex')
% end
%% Read test data
testSet = imageSet('MathWorksGallery');
size(testSet)
for j = 1:size(testSet,2)
    for i = 1:testSet(j).Count
       
       queryImage = read(testSet(j),i);
       %imshow(queryImage)
       BB= step(Facedetector, queryImage)
       mergeface = []; % to reject any un correct face detected
       if ~isempty(BB)
           if size(BB,1)>1 ; for ii=1:size(BB,1) ; mergeface = [mergeface BB(ii,2)-BB(ii,1)]; 
               end
               realfaced = find(mergeface == max(mergeface)) 
               figure
               qi = imcrop(queryImage,BB(realfaced,:));imshow(qi)
           else
               figure
               qi = imcrop(queryImage,BB(1,:));imshow(qi)
           end
           
          
%                queryFeatures = extractHOGFeatures(imresize(rgb2gray(qi),[150 150]));
%                personLabel = predict(faceClassifier,queryFeatures);
%                booleanIndex = strcmp(personLabel, personIndex);
%                integerIndex = find(booleanIndex)
%                figure;integerIndex
%                if isempty(integerIndex); title('can not facedetect');continue;else 
%                    subplot(2,1,1);imshow(queryImage);title('Query Face');
%                    subplot(2,1,2);imshow(read(faceGallery(integerIndex(1)),1));title('Matched Class');
%                end
             
           else
           continue;
       end
       
    end
end