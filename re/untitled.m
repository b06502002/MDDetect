clc; clear;
A = readmatrix('fanalysis1110_3.csv');

% heatmap(abs(A(1:200,1:200)));
imagesc(abs(A(1:303,1:226))>1);
% imagesc(abs(A(1:303,1:226))>1000);
% imagesc((abs(A(1:100,1:75))>100).*(abs(A(1:100,1:75))<1000));

