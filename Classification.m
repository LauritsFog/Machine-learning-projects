
%                           Report 2 - Classification 

%% Initialize data
clear; clc;
addpath('Data')
%Load Data table
FF_table = readtable('Alg_FFire-data.csv');
Data = table2array(FF_table(:,1:14)); % Convert table to array

% Extract column with classes and convert to 1 = "fire" and 0 = "not fire"
Classes = table2array(FF_table(:,15));
class = string(Classes);

%Fix data
Data(166,11) = 14.4;
Data(166,12) = 9;
Data(166,13) = 12.5;
Data(166,14) = 10.4;
class(166) = "fire";

for i=1:244
   
    if (class(i)=='fire')
        class(i) = '1';
    else 
        class(i) = '0';
    end
   
end
Y = str2double(class);

classNames = {'fire','not fire'}'; % Class names
C = length(classNames); % Number of classes

% Remove unvantet datapoiint (day,month, year...) 
X = Data(:,5:14);

% Split up data in the two regions
X_Bejaia = X(1:122,:);
Y_Bejaia = Y(1:122,:);

X_Sidi = X(123:244,:);
Y_Sidi = Y(123:244,:);


                            %% Region one - Bejaia
%% Two-Layer Cross Validation

y = Y_Bejaia;
x = X_Bejaia;
N = length(y);
% Create the outer folds:
K = 10;
CV = cvpartition(N, 'KFold',10);

%Initialization 

% K-nearest neighbors
Distance = 'euclidean'; % Distance measure
L = 20; % Maximum number of neighbors

% Initialize different vectors
lambda_opt_Bejaia = nan(K,1);
nabo_opt_Bejaia = nan(K,1);
Acc_Knn_opt_Bejaia = nan(K,1);
Acc_Base_opt_Bejaia = nan(K,1);
Acc_Log_opt_Bejaia = nan(K,1);

Error_log_Bejaia = nan(10,20);
Error_knn_Bejaia = nan(10,20);

% Initialize estimation vectors
y_Knn_Bejaia_est = [];
y_Base_Bejaia_est = [];
y_Log_Bejaia_est = [];
y_test_Bejaia = [];

for k = 1:K  % Outer Fold
    fprintf('Crossvalidation fold %d/%d\n', k, K);
    
    % Extract the training and test set for out fold k
    X_train = x(CV.training(k), :);
    y_train = y(CV.training(k));
    X_test = x(CV.test(k), :);
    y_test = y(CV.test(k));

    % Save the "Correct" answer 
    y_test_Bejaia = [y_test_Bejaia ; y_test];
    
    
    % Use 10-fold crossvalidation to estimate optimal values    
    KK = 10;
    CV2 = cvpartition(size(X_train,1), 'Kfold', KK);
    
for kk = 1:KK % Inner Fold
        
        % Extract training and test set
        X_train2 = X_train(CV2.training(kk), :);
        y_train2 = y_train(CV2.training(kk));
        X_test2 = X_train(CV2.test(kk), :);
        y_test2 = y_train(CV2.test(kk));
        
                                 % KNN model
                            
        for l = 1:L % For each number of neighbors
          
            % Use fitcknn to find the l nearest neighbors
            knn=fitcknn(X_train2, y_train2, 'NumNeighbors', l, 'Distance', Distance);
            y_knn_test_est2=predict(knn, X_test2);
        
            % Compute number of classification errors
            Error_knn_Bejaia(kk,l) = sum(y_test2~=y_knn_test_est2); % Count the number of errors
        end
        
        
        
                                 % Log model        
        %Initialize for log model
        mu = mean(X_train2,1);
        sigma = std(X_train2,1);
        X_train_std2 = bsxfun(@times, X_train2 - mu, 1./ sigma);
        X_test_std2 = bsxfun(@times, X_test2 - mu, 1./ sigma);
        lambda = 10.^(-5:8);
        
        for i = 1:length(lambda);
                    mdl = fitclinear(X_train_std2, y_train2, ...
                               'Lambda', lambda(i), ...
                              'Learner', 'logistic', ...
                              'Regularization', 'ridge');     
                    [y_log_test_est2, p] = predict(mdl, X_test_std2);
                    Error_log_Bejaia(kk,i) = sum(y_test2~=y_log_test_est2);
        end

end

% Train models

                             % K- Nearest Neighbour
%Define the min error and which neighbour gives that.
[min_error_Bejaia_knn, opt_neighbour] = min(sum(Error_knn_Bejaia(:,1:20))./sum(CV.TestSize)*100);    
nabo_opt_Bejaia(k) = opt_neighbour;

% Train model om optimal nabo værdi
l = opt_neighbour;
knn=fitcknn(X_train, y_train, 'NumNeighbors', l, 'Distance', Distance);
y_B_Knn_test_est=predict(knn, X_test);

s = y_B_Knn_test_est==y_test;
Acc_Knn_opt_Bejaia(k) = 100-sum(s)/numel(s)*100;
    

y_Knn_Bejaia_est = [y_Knn_Bejaia_est;y_B_Knn_test_est];


                                % Baseline model
                                
% number of 1 in training data
n1= sum(y_train); 
% number of 0 in training data
n0 = length(y_train)-n1;

y_test_est = zeros(length(y_test),1);

for i=1:length(y_test)
   
    if (n1 > n0)
        y_test_est(i) = 1;
    else 
        y_test_est(i) = 0;
    end
end

y_Base_Bejaia_est = [y_Base_Bejaia_est; y_test_est];

s = y_test_est==y_test;
Acc_Base_opt_Bejaia(k) = 100-sum(s)/numel(s)*100;

                               
                                % Logaristic Regression model

                                
[min_error_Bejaia_log, opt_lam] = min(sum(Error_log_Bejaia(:,1:20))./sum(CV.TestSize)*100); 

lambda_opt_Bejaia(k) = lambda(opt_lam);                           

mu = mean(X_train,1);
sigma = std(X_train,1);

X_train_std = bsxfun(@times, X_train - mu, 1./ sigma);
X_test_std = bsxfun(@times, X_test - mu, 1./ sigma);

mdl = fitclinear(X_train_std, y_train, ...
                 'Lambda', lambda_opt_Bejaia(k), ...
                 'Learner', 'logistic', ...
                 'Regularization', 'ridge');
[y_B_Log_test_est, p] = predict(mdl, X_test_std);
    
y_Log_Bejaia_est = [y_Log_Bejaia_est; y_B_Log_test_est];

s = y_B_Log_test_est==y_test;
Acc_Log_opt_Bejaia(k) = 100-sum(s)/numel(s)*100;

end

                            % Display accuracy

outfold = [1;2;3;4;5;6;7;8;9;10];
Table2_Bejaia = table(outfold,lambda_opt_Bejaia,Acc_Log_opt_Bejaia,nabo_opt_Bejaia,Acc_Knn_opt_Bejaia,Acc_Base_opt_Bejaia);  
 
%% Statistical evaluation

% We will use Setup I, which is the McNemera's test.

alpha = 0.05;
% Compare Baseline model and k-nearest-neighbour models:
[theta_BK_Bejaia, CI_BK_Bejaia, p_BK_Bejaia] = mcnemar(y_test_Bejaia, ... 
y_Knn_Bejaia_est, y_Base_Bejaia_est, alpha);

% Compare Baseline model and Log models:
[theta_BL_Bejaia, CI_BL_Bejaia, p_BL_Bejaia] = mcnemar(y_test_Bejaia, ... 
y_Log_Bejaia_est, y_Base_Bejaia_est, alpha);

% Compare k-nearest-neighbour model and Log models:
[theta_KL_Bejaia, CI_KL_Bejaia, p_KL_Bejaia] = mcnemar(y_test_Bejaia, ... 
y_Log_Bejaia_est, y_Knn_Bejaia_est, alpha);

                            % Create table 

Comparison = ["Knn/Base";"Log/Base"; "Log/Knn"];
Theta = [theta_BK_Bejaia; theta_BL_Bejaia; theta_KL_Bejaia];

LowerBound = [CI_BK_Bejaia(1);CI_BL_Bejaia(1);CI_KL_Bejaia(1)];
UpperBound = [CI_BK_Bejaia(2);CI_BL_Bejaia(2);CI_KL_Bejaia(2)];
ConfidenceInterval = table(LowerBound,UpperBound);

pValue = [p_BK_Bejaia; p_BL_Bejaia; p_KL_Bejaia];

Table_Bejaia = table(Theta,ConfidenceInterval,pValue, 'RowNames', Comparison);

% Det er i hvert fald evidens for at Log er bedre end Base og at Knn er
% bedre.

% Den siger måske også at Log er bedre end Knn vel??

                            






                            %% Region two - Sidi
%% Two-Layer Cross Validation

y = Y_Sidi;
x = X_Sidi;
N = length(y);
% Create the outer folds:
K = 10;
CV = cvpartition(N, 'KFold',10);

%Initialization 

% K-nearest neighbors
Distance = 'euclidean'; % Distance measure
L = 20; % Maximum number of neighbors

% Initialize different vectors
lambda_opt_Sidi = nan(K,1);
nabo_opt_Sidi = nan(K,1);
Acc_Knn_opt_Sidi = nan(K,1);
Acc_Base_opt_Sidi = nan(K,1);
Acc_Log_opt_Sidi = nan(K,1);

Error_log_Sidi = nan(10,20);
Error_knn_Sidi = nan(10,20);

% Initialize estimation vectors
y_Knn_Sidi_est = [];
y_Base_Sidi_est = [];
y_Log_Sidi_est = [];
y_test_Sidi = [];

for k = 1:K  % Outer Fold
    fprintf('Crossvalidation fold %d/%d\n', k, K);
    
    % Extract the training and test set for out fold k
    X_train = x(CV.training(k), :);
    y_train = y(CV.training(k));
    X_test = x(CV.test(k), :);
    y_test = y(CV.test(k));

    % Save the "Correct" answer 
    y_test_Sidi = [y_test_Sidi ; y_test];
    
    
    % Use 10-fold crossvalidation to estimate optimal values    
    KK = 10;
    CV2 = cvpartition(size(X_train,1), 'Kfold', KK);
    
for kk = 1:KK % Inner Fold
        
        % Extract training and test set
        X_train2 = X_train(CV2.training(kk), :);
        y_train2 = y_train(CV2.training(kk));
        X_test2 = X_train(CV2.test(kk), :);
        y_test2 = y_train(CV2.test(kk));
        
                                 % KNN model
                            
        for l = 1:L % For each number of neighbors
          
            % Use fitcknn to find the l nearest neighbors
            knn=fitcknn(X_train2, y_train2, 'NumNeighbors', l, 'Distance', Distance);
            y_knn_test_est2=predict(knn, X_test2);
        
            % Compute number of classification errors
            Error_knn_Sidi(kk,l) = sum(y_test2~=y_knn_test_est2); % Count the number of errors
        end
        
        
        
                                 % Log model        
        %Initialize for log model
        mu = mean(X_train2,1);
        sigma = std(X_train2,1);
        X_train_std2 = bsxfun(@times, X_train2 - mu, 1./ sigma);
        X_test_std2 = bsxfun(@times, X_test2 - mu, 1./ sigma);
        lambda = 10.^(-5:8);
        
        for i = 1:length(lambda);
                    mdl = fitclinear(X_train_std2, y_train2, ...
                               'Lambda', lambda(i), ...
                              'Learner', 'logistic', ...
                              'Regularization', 'ridge');     
                    [y_log_test_est2, p] = predict(mdl, X_test_std2);
                    Error_log_Sidi(kk,i) = sum(y_test2~=y_log_test_est2);
        end

end

% Train models

                             % K- Nearest Neighbour
%Define the min error and which neighbour gives that.
[min_error_Sidi_knn, opt_neighbour] = min(sum(Error_knn_Sidi(:,1:20))./sum(CV.TestSize)*100);    
nabo_opt_Sidi(k) = opt_neighbour;

% Train model om optimal nabo værdi
l = opt_neighbour;
knn=fitcknn(X_train, y_train, 'NumNeighbors', l, 'Distance', Distance);
y_Sidi_Knn_test_est=predict(knn, X_test);

s = y_Sidi_Knn_test_est==y_test;
Acc_Knn_opt_Sidi(k) = 100-sum(s)/numel(s)*100;
    

y_Knn_Sidi_est = [y_Knn_Sidi_est;y_Sidi_Knn_test_est];


                                % Baseline model
                                
% number of 1 in training data
n1= sum(y_train); 
% number of 0 in training data
n0 = length(y_train)-n1;

y_test_est = zeros(length(y_test),1);

for i=1:length(y_test)
   
    if (n1 > n0)
        y_test_est(i) = 1;
    else 
        y_test_est(i) = 0;
    end
end

y_Base_Sidi_est = [y_Base_Sidi_est; y_test_est];

s = y_test_est==y_test;
Acc_Base_opt_Sidi(k) = 100-sum(s)/numel(s)*100;

                               
                                % Logaristic Regression model

                                
[min_error_Sidi_log, opt_lam] = min(sum(Error_log_Sidi(:,1:20))./sum(CV.TestSize)*100); 

lambda_opt_Sidi(k) = lambda(opt_lam);                           

mu = mean(X_train,1);
sigma = std(X_train,1);

X_train_std = bsxfun(@times, X_train - mu, 1./ sigma);
X_test_std = bsxfun(@times, X_test - mu, 1./ sigma);

mdl = fitclinear(X_train_std, y_train, ...
                 'Lambda', lambda_opt_Sidi(k), ...
                 'Learner', 'logistic', ...
                 'Regularization', 'ridge');
[y_Sidi_Log_test_est, p] = predict(mdl, X_test_std);
    
y_Log_Sidi_est = [y_Log_Sidi_est; y_Sidi_Log_test_est];

s = y_Sidi_Log_test_est==y_test;
Acc_Log_opt_Sidi(k) = 100-sum(s)/numel(s)*100;

end

                            % Display accuracy

outfold = [1;2;3;4;5;6;7;8;9;10];
Table2_Sidi = table(outfold,lambda_opt_Sidi,Acc_Log_opt_Sidi,nabo_opt_Sidi,Acc_Knn_opt_Sidi,Acc_Base_opt_Sidi); 
 
%% Statistical evaluation

% We will use Setup I, which is the McNemera's test.

alpha = 0.05;
% Compare Baseline model and k-nearest-neighbour models:
[theta_BK_Sidi, CI_BK_Sidi, p_BK_Sidi] = mcnemar(y_test_Sidi, ... 
y_Knn_Sidi_est, y_Base_Sidi_est, alpha);

% Compare Baseline model and Log models:
[theta_BL_Sidi, CI_BL_Sidi, p_BL_Sidi] = mcnemar(y_test_Sidi, ... 
y_Log_Sidi_est, y_Base_Sidi_est, alpha);

% Compare k-nearest-neighbour model and Log models:
[theta_KL_Sidi, CI_KL_Sidi, p_KL_Sidi] = mcnemar(y_test_Sidi, ... 
y_Log_Sidi_est, y_Knn_Sidi_est, alpha);

                            % Create table 

Comparison = ["Knn/Base";"Log/Base"; "Log/Knn"];
Theta = [theta_BK_Sidi; theta_BL_Sidi; theta_KL_Sidi];

LowerBound = [CI_BK_Sidi(1);CI_BL_Sidi(1);CI_KL_Sidi(1)];
UpperBound = [CI_BK_Sidi(2);CI_BL_Sidi(2);CI_KL_Sidi(2)];
ConfidenceInterval = table(LowerBound,UpperBound);

pValue = [p_BK_Sidi; p_BL_Sidi; p_KL_Sidi];

Table_Sidi = table(Theta,ConfidenceInterval,pValue, 'RowNames', Comparison);

% Det er i hvert fald evidens for at Log er bedre end Base og at Knn er
% bedre.

% Den siger måske også at Log er bedre end Knn vel??







                         %% The two regions together

%% Two-Layer Cross Validation

y = Y;
x = X;
N = length(y);
% Create the outer folds:
K = 10;
CV = cvpartition(N, 'KFold',10);

%Initialization 

% K-nearest neighbors
Distance = 'euclidean'; % Distance measure
L = 20; % Maximum number of neighbors

% Initialize different vectors
lambda_opt_T = nan(K,1);
nabo_opt_T = nan(K,1);
Acc_Knn_opt_T = nan(K,1);
Acc_Base_opt_T = nan(K,1);
Acc_Log_opt_T = nan(K,1);

Error_log_T = nan(10,20);
Error_knn_T = nan(10,20);

% Initialize estimation vectors
y_Knn_T_est = [];
y_Base_T_est = [];
y_Log_T_est = [];
y_test_T = [];

for k = 1:K  % Outer Fold
    fprintf('Crossvalidation fold %d/%d\n', k, K);
    
    % Extract the training and test set for out fold k
    X_train = x(CV.training(k), :);
    y_train = y(CV.training(k));
    X_test = x(CV.test(k), :);
    y_test = y(CV.test(k));

    % Save the "Correct" answer 
    y_test_T = [y_test_T ; y_test];
    
    
    % Use 10-fold crossvalidation to estimate optimal values    
    KK = 10;
    CV2 = cvpartition(size(X_train,1), 'Kfold', KK);
    
for kk = 1:KK % Inner Fold
        
        % Extract training and test set
        X_train2 = X_train(CV2.training(kk), :);
        y_train2 = y_train(CV2.training(kk));
        X_test2 = X_train(CV2.test(kk), :);
        y_test2 = y_train(CV2.test(kk));
        
                                 % KNN model
                            
        for l = 1:L % For each number of neighbors
          
            % Use fitcknn to find the l nearest neighbors
            knn=fitcknn(X_train2, y_train2, 'NumNeighbors', l, 'Distance', Distance);
            y_knn_test_est2=predict(knn, X_test2);
        
            % Compute number of classification errors
            Error_knn_T(kk,l) = sum(y_test2~=y_knn_test_est2); % Count the number of errors
        end
        
        
        
                                 % Log model        
        %Initialize for log model
        mu = mean(X_train2,1);
        sigma = std(X_train2,1);
        X_train_std2 = bsxfun(@times, X_train2 - mu, 1./ sigma);
        X_test_std2 = bsxfun(@times, X_test2 - mu, 1./ sigma);
        lambda = 10.^(-5:8);
        
        for i = 1:length(lambda);
                    mdl = fitclinear(X_train_std2, y_train2, ...
                               'Lambda', lambda(i), ...
                              'Learner', 'logistic', ...
                              'Regularization', 'ridge');     
                    [y_log_test_est2, p] = predict(mdl, X_test_std2);
                    Error_log_T(kk,i) = sum(y_test2~=y_log_test_est2);
        end

end

% Train models

                             % K- Nearest Neighbour
%Define the min error and which neighbour gives that.
[min_error_T_knn, opt_neighbour] = min(sum(Error_knn_T(:,1:20))./sum(CV.TestSize)*100);    
nabo_opt_T(k) = opt_neighbour;

% Train model om optimal nabo værdi
l = opt_neighbour;
knn=fitcknn(X_train, y_train, 'NumNeighbors', l, 'Distance', Distance);
y_T_Knn_test_est=predict(knn, X_test);

s = y_T_Knn_test_est==y_test;
Acc_Knn_opt_T(k) = 100-sum(s)/numel(s)*100;
    

y_Knn_T_est = [y_Knn_T_est;y_T_Knn_test_est];


                                % Baseline model
                                
% number of 1 in training data
n1= sum(y_train); 
% number of 0 in training data
n0 = length(y_train)-n1;

y_test_est = zeros(length(y_test),1);

for i=1:length(y_test)
   
    if (n1 > n0)
        y_test_est(i) = 1;
    else 
        y_test_est(i) = 0;
    end
end

y_Base_T_est = [y_Base_T_est; y_test_est];

s = y_test_est==y_test;
Acc_Base_opt_T(k) = 100-sum(s)/numel(s)*100;

                               
                                % Logaristic Regression model

                                
[min_error_T_log, opt_lam] = min(sum(Error_log_T(:,1:20))./sum(CV.TestSize)*100); 

lambda_opt_T(k) = lambda(opt_lam);                           

mu = mean(X_train,1);
sigma = std(X_train,1);

X_train_std = bsxfun(@times, X_train - mu, 1./ sigma);
X_test_std = bsxfun(@times, X_test - mu, 1./ sigma);

mdl = fitclinear(X_train_std, y_train, ...
                 'Lambda', lambda_opt_T(k), ...
                 'Learner', 'logistic', ...
                 'Regularization', 'ridge');
[y_T_Log_test_est, p] = predict(mdl, X_test_std);
    
y_Log_T_est = [y_Log_T_est; y_T_Log_test_est];

s = y_T_Log_test_est==y_test;
Acc_Log_opt_T(k) = 100-sum(s)/numel(s)*100;

end

                            % Display accuracy

outfold = [1;2;3;4;5;6;7;8;9;10];
Table2_Together = table(outfold,lambda_opt_T,Acc_Log_opt_T,nabo_opt_T,Acc_Knn_opt_T,Acc_Base_opt_T); 
 
%% Statistical evaluation

% We will use Setup I, which is the McNemera's test.

alpha = 0.05;
% Compare Baseline model and k-nearest-neighbour models:
[theta_BK_T, CI_BK_T, p_BK_T] = mcnemar(y_test_T, ... 
y_Knn_T_est, y_Base_T_est, alpha);

% Compare Baseline model and Log models:
[theta_BL_T, CI_BL_T, p_BL_T] = mcnemar(y_test_T, ... 
y_Log_T_est, y_Base_T_est, alpha);

% Compare k-nearest-neighbour model and Log models:
[theta_KL_T, CI_KL_T, p_KL_T] = mcnemar(y_test_T, ... 
y_Log_T_est, y_Knn_T_est, alpha);

                            % Create table 

Comparison = ["Knn/Base";"Log/Base"; "Log/Knn"];
Theta = [theta_BK_T; theta_BL_T; theta_KL_T];

LowerBound = [CI_BK_T(1);CI_BL_T(1);CI_KL_T(1)];
UpperBound = [CI_BK_T(2);CI_BL_T(2);CI_KL_T(2)];
ConfidenceInterval = table(LowerBound,UpperBound);

pValue = [p_BK_T; p_BL_T; p_KL_T];

Table_Together = table(Theta,ConfidenceInterval,pValue, 'RowNames', Comparison);

% Det er i hvert fald evidens for at Log er bedre end Base og at Knn er
% bedre.

% Den siger måske også at Log er bedre end Knn vel??

%% Tables

Table2_Bejaia
Table_Bejaia

Table2_Sidi
Table_Sidi

Table2_Together
Table_Together

%% Logistic Regression model

x = X;
y = Y;

CV = cvpartition(classNames(y+1), 'Holdout', .20);

% Extract the training and test set
X_train = x(CV.training, :);
y_train = y(CV.training);
X_test = x(CV.test, :);
y_test = y(CV.test);

% Standardize the data
mu = mean(X_train,1);
sigma = std(X_train,1);
X_train_std = bsxfun(@times, X_train - mu, 1./ sigma);
X_test_std = bsxfun(@times, X_test - mu, 1./ sigma);
lam = 10.^-5;

mdl = fitclinear(X_train_std, y_train, ...
                 'Lambda', lam, ...
                 'Learner', 'logistic', ...
                 'Regularization', 'ridge');
    [y_train_est, p] = predict(mdl, X_train_std);
    train_error = sum( y_train ~= y_train_est ) / length(y_train);
    
    [y_test_est, p] = predict(mdl, X_test_std);
    test_error = sum( y_test ~= y_test_est ) / length(y_test);
    
    coefficient_norm = norm(mdl.Beta,2);

s = y_test_est==y_test;
Acc = 100-sum(s)/numel(s)*100;

weight = mdl.Beta;



