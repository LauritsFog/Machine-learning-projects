function rate = nc_err_frac(Wi,Wo,Inputs,Targets) 

%NC_ERR_FRAC - Neural classifier fraction of misclassified examples
%
% rate = nc_err_frac(Wi,Wo,Inputs,Targets)
%
% Input:
%    Wi      : Matrix with input-to-hidden weights
%    Wo      : Matrix with hidden-to-outputs weights
%    Inputs  : Matrix with examples as rows
%    Targets : Matrix with target values as rows
%
% Output:
%    rate    : The fraction of misclassified examples
%
% Binary Neural Classifier, version 1.0 
% Sigurdur Sigurdsson 2002, DSP, IMM, DTU.

% Determine the number of examples and classes
N = size(Inputs,1);
  
% Calculate the class probabilities without outlier probability
[Vj,probs] = nc_forward(Wi,Wo,Inputs);
probs = nc_logistic(probs);
  
% Choose the target class boundary as 0.5
t_est = round(probs);

% Compute the misclassification rate
rate = sum(Targets ~= t_est)/N;
