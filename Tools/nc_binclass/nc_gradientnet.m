function [grad_Wi,grad_Wo] = nc_gradientnet(Wi,Wo,Inputs,Targets)

% NC_GRADIENTNET - Partial derivative of network output before softmax  
%                  w.r.t. weights for all examples
%
% [grad_Wi,grad_Wo] = nc_gradientnet(Wi,Wo,Inputs,Targets)
%
% Input:
%    Wi      : Matrix with input-to-hidden weights
%    Wo      : Matrix with hidden-to-outputs weights
%    Inputs  : Matrix with examples as rows
%    Targets : Matrix with target values as rows
%
% Outputs:
%    grad_Wi : Matrix with gradient for input weights
%    grad_Wo : Matrix with gradient for output weights
%
% Binary Neural Classifier, version 1.0 
% Sigurdur Sigurdsson 2002, DSP, IMM, DTU.

% Determine the number of examples
[exam inp] = size(Inputs);
 
% Calculate hidden and output unit activations
[Vj,yj] = nc_forward(Wi,Wo,Inputs);

% Compute the gradient for the hidden-to-output weights for each example
grad_Wo = [Vj ones(exam,1)];
    
% Compute the gradient for the input-to-hidden weights for each example
hidden_part = repmat((1.0-Vj.^2),1,size(Wi,2));
input_part = (kron([Inputs ones(exam,1)],ones(1,size(Wi,1))));
output_part = repmat(Wo(1,1:end-1),exam,size(Wi,2));
grad_Wi = hidden_part.*input_part.*output_part;
