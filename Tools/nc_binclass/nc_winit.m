function [Wi,Wo] = nc_winit(Ni,Nh)

%NC_WINIT - Initialize weights in the network with 
%           a Gaussian distribution N(0,1/alpha)
%
% [Wi,Wo] = NR_WINIT(Ni,Nh)
%
% Input:
%    Ni    : Number of input neurons
%    Nh    : Number of hidden neurons
%
% Output:
%    Wi    : Input-to-hidden initial weights
%    Wo    : Hidden-to-output initial weights
%
% Binary Neural Classifier, version 1.0 
% Sigurdur Sigurdsson 2002, DSP, IMM, DTU.
  
Wi = randn(Nh,Ni+1)/sqrt(Ni);
Wo = randn(1,Nh+1)/sqrt(Ni);
