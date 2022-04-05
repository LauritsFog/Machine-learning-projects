function [alpha,gamma] = nc_alpha(W,par)

%NC_ALPHA - Compute the regularization parameter with
%           MacKay's MLII scheme and the number of well 
%           determined weights 
%
% [alpha,gamma] = nc_alpha(W,par)
%  
% Input:
%    W     : The weight vector
%    par   : Struct variable with a number of parameters
%
% Output:
%    alpha : The regularization parameter
%    gamma : The number of well determined weights
%
% Binary Neural Classifier, version 1.0 
% Sigurdur Sigurdsson 2002, DSP, IMM, DTU.

% Transform the weights from a weight vector to matrices
[Wi,Wo] = nc_W2Wio(W,par.Ni,par.Nh);

% Change variable names
Inputs = par.x;
Targets = par.t;
alpha = par.alpha;

% Get the number of weights
Nw = length(W);

% Estimate the alpha parameter and number of effective weights
Ew = 0.5*sum(W.^2);
A = nc_gnhess(Wi,Wo,alpha,Inputs,Targets);
gamma = Nw-alpha*trace(inv(A));
alpha = gamma/(2*Ew);
