function [thetahat, CI, p] = mcnemar(y_true, yhatA, yhatB, alpha, verbose)
if nargin < 4
    alpha = 0.05; 
end
if nargin < 5
    verbose = false;
end
%% perform McNemars test
n = zeros(2);
c1 = yhatA == y_true;
c2 = yhatB == y_true;
nn(1,1) = sum(c1 & c2);
nn(1,2) = sum(c1 & ~c2);
nn(2,1) = sum(~c1 & c2);
nn(2,2) = sum(~c1 & ~c2);
n = sum(nn(:));

Etheta = (nn(1,2)-nn(2,1))/n;
n12 = nn(1,2); n21 = nn(2,1);
Q = n^2 * (n+1) * (Etheta+1) * (1-Etheta) / ( (n*(n12+n21) - (n12-n21)^2) );
p = (Etheta + 1)/2 * (Q-1);
q = (1-Etheta)/2 * (Q-1);
thetaL = betainv(alpha/2, p, q)*2-1;
thetaU = betainv(1-alpha/2, p, q)*2-1;
%%
CI = [thetaL,thetaU];
p = 2*binocdf(min([n12,n21]), n12+n21, 0.5);
if verbose
    disp("Result of McNemars test using alpha=" + alpha)
    disp("Comparison matrix n")
    disp(nn)
    if n12+n21 <= 10
        disp("Warning, n12+n21 is low: n12+n21=" + (n12+n21))
    end
    disp("Approximate 1-alpha confidence interval of theta: [thetaL,thetaU] = [" + CI(1) + ", " + CI(2) +"]");
    disp("p-value for two-sided test A and B have same accuracy (exact binomial test): p=" + p)
end

thetahat = (nn(1,2)-nn(2,1))/n;
end
