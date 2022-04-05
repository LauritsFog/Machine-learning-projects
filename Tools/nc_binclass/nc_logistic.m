function probs = nc_logistic(phi)

%NC_LOGISTIC - Neural classifier logistic sigmoidal function
%
% probs = nc_logistic(phi)
%
% Input:  
%    phi  : Vector of outputs of the network from NC_FORWARD
%           where rows are the individual output neurons.
% Output:  
%    probs: Vector of posterior probabilities. Each row is the
%           class probability for class with target value 1
%           for a specific example.
% 
% Binary Neural Classifier, version 1.0 
% Sigurdur Sigurdsson 2002, DSP, IMM, DTU.

% Number of classes
Nc = size(phi,2);

% Compute the class probabilities
probs = 1./(1+exp(-phi));