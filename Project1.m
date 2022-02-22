%%
%Getting the file path to where the data is located. 
cdir = fileparts(mfilename('fullpath'));
%Located in a subfolder Called Data
file_path = fullfile(cdir,'/Data/Alg_FFire-data.csv');
%Reading the data from the csv file to matlab
%The data splits into a new region at ID 123 read the Readme file in the
%Data folder for more information.
FF_table = readtable(file_path);

%%

% Adding Data directory to path. 
addpath(genpath('Data'));

% Reading the csv. file into a table. 
FF_table = readtable('Alg_FFire-data.csv');

%% Principal component analysis: 

% Not considering date and class for the SVD. 
A = table2array(FF_table(:,5:14));

% Removing row with NaN's and missing classification. 
A(166,:) = [];

% Subtracting the mean. 
A = A - mean(A);

% Computing the SVD. 
[U,S,V] = svd(A);

% Extracting diagonal elements from S. 
sigmas = diag(S);

explainedVar = zeros(1,10);

% Computing explained variance for each principle component. 
for i = 1:10
    explainedVar(i) = sigmas(i)^2/sum(sigmas.^2);
end

explainedVarCum = zeros(1,10);

% Computing cumulative explained variance for each principle component. 

for i = 1:10
    explainedVarCum(i) = sum(sigmas(1:i).^2)/sum(sigmas.^2); 
end

%%

plot(explainedVarCum*100);

