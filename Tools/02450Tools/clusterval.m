function [Rand, Jaccard, NMI] = clusterval(y, i)
% CLUSTERVAL Estimate cluster validity using entroy, purity, rand index,
% and Jaccard coefficient.
%
% Usage:
%   [Rand, Jaccard, NMI] = clusterval(y, i);
%
% Input:
%    y         N-by-1 vector of class labels 
%    i         N-by-1 vector of cluster indices
%
% Output:
%   Rand       Rand index.
%   Jaccard    Jaccard coefficient.
%   NMI        Normalized mutual information
% Copyright 2011, Mikkel N. Schmidt, Morten Mørup, Technical University of Denmark
% Updated 3 November 2015 to not count self pairs in Rand and Jaccard
% Updated 17 October 2016 to include NMI replacing two other measures


N = length(y);
[d1_,d2_,jy] = unique(y);
Zy = sparse(jy, 1:N, ones(1,N));
Ay = Zy'*Zy;

[d1_,d2_,ji] = unique(i);
Zi = sparse(ji, 1:N, ones(1,N));
Ai = Zi'*Zi;

f11 = full(sum(sum(triu(Ai.*Ay,1))));
f00 = full(sum(sum(triu((1-Ai).*(1-Ay),1))));

Rand = (f11+f00)/(N*(N-1)/2);
Jaccard = f11/(N*(N-1)/2-f00);
NMI=full((calcMI(Zy,Zi))/(sqrt(calcMI(Zy,Zy)*calcMI(Zi,Zi))));



