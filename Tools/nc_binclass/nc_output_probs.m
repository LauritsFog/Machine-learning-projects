function [p_est,Targets_est] = nc_output_probs(Wi,Wo,Inputs)

%NC_OUTPUT_PROBS - Compute estimated class probabilities and target label 
%
% [p_est,Targets_est] = nc_output_probs(Wi,Wo,Inputs)
%
% Inputs:
%   Wi          : The input-to-hidden weight matrix
%   Wo          : The hidden-to-output weight matrix
%   Inputs      : The input data matrix
%
% Output:
%   p_est       : The estimated class probability
%   Targets_est : The estimated target labels
%
% Binary Neural Classifier, version 1.0 
% Sigurdur Sigurdsson 2002, DSP, IMM, DTU.

% Calculate the estimated posterior for all examples and both classes 
[Vj,yj] = nc_forward(Wi,Wo,Inputs);
p_est = nc_logistic(yj);

% Calculate the estimated posterior for all examples and classes 
Targets_est = round(p_est);