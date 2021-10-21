clc; clear;
A = readmatrix('fanalysis.csv');

% heatmap(abs(A(1:200,1:200)));
% imagesc(abs(A)>20);
% imagesc(abs(A(1:303,1:226))>1000);
imagesc((abs(A(1:100,1:75))>100).*(abs(A(1:100,1:75))<1000));

