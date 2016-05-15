%read the image
A=imread('DSCN1199.JPG');
%Get face detector object
Facedetector = vision.CascadeObjectDetector();
BBox= step(Facedetector, A)
CC = imcrop(A,BBox)
imshow(CC)
% use face detector on A and get the the faces
B = insertObjectAnnotation(A,'rectangle',BBox,'Ahmed'); %annotation(comment)
imshow(B),title('detected faces');
%display the number of faces
n = size(BBox,1);
str = strcat('number of faces are ', char(n));
disp(str);