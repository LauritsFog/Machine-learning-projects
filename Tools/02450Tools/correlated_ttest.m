function [p,CI] = correlated_ttest(r, rho, alpha)
% Implement the corrected t-test. 
% The parameter rho is usually 1/K 
% where K is the number of folds used in cross-validation
% (see lecture notes)
if nargin < 3
    alpha = 0.05; 
end
rhat = mean(r);
shat = std(r);
J = length(r); 
sigmatilde = shat * sqrt(1/J + rho / (1-rho) );
CI = tinv([alpha/2, 1-alpha/2], J-1) * sigmatilde + rhat;
th = rhat / sigmatilde;
p = tcdf(-abs(th), J-1);
end