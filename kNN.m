clc;
close all;

X_train = importdata('VidTIMIT/X_train.mat');
y_train = importdata('VidTIMIT/y_train.mat');
X_test = importdata('VidTIMIT/X_test.mat');

k = 5;

%classify KNN%
model = fitcknn(X_train,y_train,'NumNeighbors',k,'Distance','euclidean');
label = predict(model,X_test);

y_test = importdata('VidTIMIT/y_test.mat');

%Measuring accuracy%
accuracy_KNN = classperf(y_test,label);
fprintf('KNN for VidTIMIT, Accuracy = %.2f%%\n',accuracy_KNN.CorrectRate*100);