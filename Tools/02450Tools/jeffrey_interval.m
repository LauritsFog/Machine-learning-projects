function [thetahat, CI] = jeffrey_interval(y_true, yhat, alpha)
if nargin < 3
    alpha = 0.05;
end

m = sum(y_true == yhat);
n = length(y_true);

a = m+.5; 
b = n-m+.5; 

CI = betainv([alpha/2, 1-alpha/2], a, b);
if m == 0
    CI(1) = 0;
end
if m == n
    CI(2) = 1;
end
thetahat = a/(a+b);

end