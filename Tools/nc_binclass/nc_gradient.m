function [dWi,dWo] = nc_gradient(Wi,Wo,alpha,Inputs,Targets)

%NC_GRADIENT   Neural classifier gradient of the cost function 
%              with respect to the weights
%
% [dWi,dWo] = nc_gradient(Wi,Wo,alpha,Inputs,Targets)
%
% Input:
%    Wi      : Matrix with input-to-hidden weights
%    Wo      : Matrix with hidden-to-outputs weights
%    alpha   : Weight decay parameter
%    Inputs  : Matrix with examples as rows
%    Targets : Matrix with target values as rows
%
% Output:
%    dWi     : Matrix with gradient for input weights
%    dWo     : Matrix with gradient for output weights
%
% Binary Neural Classifier, version 1.0 
% Sigurdur Sigurdsson 2002, DSP, IMM, DTU.

% Determine the number of examples, classes and hidden units
[exam inp] = size(Inputs);
Nh = size(Wi,1);
 
% Calculate hidden and output unit activations
[Vj,yj] = nc_forward(Wi,Wo,Inputs);

% Apply logistic sigmoidal
yj = nc_logistic(yj);

% Output unit deltas
delta_o = (yj-Targets);

% Hidden unit deltas
delta_h = (1.0 - Vj.^2) .* (delta_o * Wo(:,1:Nh));
  
% Partial derivatives for the output weights
dWo = delta_o' * [Vj ones(exam,1)];

% Partial derivatives for the input weights
dWi = delta_h' * [Inputs ones(exam,1)];
  
% Add derivatives of the weight decay term
dWi = dWi + alpha*Wi;
dWo = dWo + alpha*Wo;