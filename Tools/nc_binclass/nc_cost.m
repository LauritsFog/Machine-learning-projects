function cost = nc_cost(Wi,Wo,alpha,Inputs,Targets) 

%NC_COST - Neural classifier costfunction with regularization
%   
% cost = nc_cost(Wi,Wo,alpha,Inputs,Targets)
%
% Input:
%    Wi      : Matrix with input-to-hidden weights
%    Wo      : Matrix with hidden-to-outputs weights
%    alpha:  : Weight decay parameter
%    Inputs  : Matrix with examples as rows
%    Targets : Matrix with target values as rows
%
% Output:
%    cost    : Value of regularized cost function
%
% Binary Neural Classifier, version 1.0 
% Sigurdur Sigurdsson 2002, DSP, IMM, DTU.

% Calculate the estimated posterior for all examples and classes
[Vj,yj] = nc_forward(Wi,Wo,Inputs);
Targets_est = nc_logistic(yj);

% Determine the number of classes and examples
N = length(Targets);

% Compute the likelihood error
cost = -sum([log(Targets_est(Targets==1));log(1-Targets_est(Targets==0))]);

% Add the regularization term to give the error
cost = cost + 0.5*alpha*(sum(sum(Wi.^2)) + sum(sum(Wo.^2))); 
