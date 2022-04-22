function H = nc_gnhess(Wi,Wo,alpha,Inputs,Targets)

% NC_GNHESS - Evaluate the Gauss-Newton Hessian matrix of the cost function 
% 
% A = nc_gnhess(Wi,Wo,alpha,Inputs,Targets)
%
% Inputs:
%    Wi      : The input-to-hidden weight matrix
%    Wo      : The hidden-to-output weight matrix
%    alpha   : The regularization parameter
%    Inputs  : The input data matrix
%    Targets : The target labels
%
% Output:
%    A       : The Gauss-Newton approximated Hessian matrix
%
% Binary Neural Classifier, version 1.0 
% Sigurdur Sigurdsson 2002, DSP, IMM, DTU.

% Gradient of the outputs w.r.t. the inputs of the network 
[grad_Wi,grad_Wo] = nc_gradientnet(Wi,Wo,Inputs,Targets);
g = [grad_Wi grad_Wo];

% Determine the number of weights and examples
Nw = size(g,2);
%N = size(Inputs,1);

% Compute the GN-Hessian 
H = g'*g;

% Add the regularization term
H = H + alpha*eye(Nw);