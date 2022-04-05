load_data;

addpath(genpath('../Tools/nr'));

attributeNames = FF_table(:,5:14).Properties.VariableNames;

% Extracting X and Y data. X = RH, Ws, Rain and Y = temp. 
X = table2array(FF_table(:,6:8));
y = table2array(FF_table(:,5));

X = [ones(size(X,1),1) X];

% Removing row with NaN's and missing classification. 
X(166,:) = [];
y(166,:) = [];

%%
M=3+1;
N = length(y);
attributeNames={'Offset', attributeNames{1:end}};
 
% Crossvalidation
% Create crossvalidation partition for evaluation of performance of optimal
% model
K = 10;
CV = cvpartition(N, 'Kfold', K);

% Values of lambda
lambda_tmp=10.^(-5:8);

% Initialize variables
T=length(lambda_tmp);
Error_train_rlr = nan(K,1);
Error_test_rlr = nan(K,1);
Error_train_nofeatures = nan(K,1);
Error_test_nofeatures = nan(K,1);
Error_train2 = nan(T,K);
Error_test2 = nan(T,K);
w = nan(M,T,K);
lambda_opt = nan(K,1);
w_rlr = nan(M,K);
mu = nan(K, M-1);
sigma = nan(K, M-1);
w_noreg = nan(M,K);
lineRegEgens = zeros(K,T);

NHiddenUnits = [1:10];
Error_train_ANN = nan(K,1);
Error_test_ANN = nan(K,1);
Error_train2_ANN = nan(length(NHiddenUnits),K);
Error_test2_ANN = nan(length(NHiddenUnits),K);
bestnet=cell(K,1);
h_opt = nan(K,1);
ANNEgens = zeros(K,K);

% For each crossvalidation fold
for k = 1:K
    fprintf('Crossvalidation fold %d/%d\n', k, K);
    
    % Extract the training and test set
    X_train = X(CV.training(k), :);
    y_train = y(CV.training(k));
    X_test = X(CV.test(k), :);
    y_test = y(CV.test(k));

    % Use 10-fold crossvalidation to estimate optimal value of lambda    
    KK = 10;
    CV2 = cvpartition(size(X_train,1), 'Kfold', KK);
    for kk=1:KK
        X_train2 = X_train(CV2.training(kk), :);
        y_train2 = y_train(CV2.training(kk));
        X_test2 = X_train(CV2.test(kk), :);
        y_test2 = y_train(CV2.test(kk));
        
        % Standardize the training and test set based on training set in
        % the inner fold
        mu2 = mean(X_train2(:,2:end));
        sigma2 = std(X_train2(:,2:end));
        X_train2_std = X_train2;
        X_test2_std = X_test2;
        X_train2_std(:,2:end) = (X_train2(:,2:end) - mu2) ./ sigma2;
        X_test2_std(:,2:end) = (X_test2(:,2:end) - mu2) ./ sigma2;
        
        Xty2 = X_train2_std' * y_train2;
        XtX2 = X_train2_std' * X_train2_std;
        for t=1:length(lambda_tmp)   
            % Learn parameter for current value of lambda for the given
            % inner CV_fold
            regularization = lambda_tmp(t) * eye(M);
            regularization(1,1) = 0; % Remove regularization of bias-term
            w(:,t,kk)=(XtX2+regularization)\Xty2;
            % Evaluate training and test performance
            Error_train2(t,kk) = sum((y_train2-X_train2_std*w(:,t,kk)).^2)/CV2.TrainSize(kk);
            Error_test2(t,kk) = sum((y_test2-X_test2_std*w(:,t,kk)).^2)/CV2.TestSize(kk);
        end
        
        MSEBest = inf;
        for t = 1:length(NHiddenUnits)
            netwrk = nr_main(X_train2(:,2:end), y_train2, X_test2(:,2:end), y_test2, NHiddenUnits(t));
            if netwrk.mse_train(end) < MSEBest 
                bestnet{kk} = netwrk;
                MSEBest=netwrk.mse_train(end);
            end
            y_train_est = netwrk.t_pred_train;    
            y_test_est = netwrk.t_pred_test;        

            % Compute least squares error
            Error_train2_ANN(t,kk) = sum((y_train2-y_train_est).^2)/CV2.TrainSize(kk);
            Error_test2_ANN(t,kk) = sum((y_test2-y_test_est).^2)/CV2.TestSize(kk); 
        end
    end
    
    for s = 1:length(NHiddenUnits)
        ANNEgens(k,s) = Error_test2_ANN(s,:)*CV2.TestSize'/CV.TrainSize(k);
    end
    for s = 1:length(lambda_tmp)
        lineRegEgens(k,s) = Error_test2(s,:)*CV2.TestSize'/CV.TrainSize(k);
    end
    
    [val_ANN,ind_opt_ANN]=min(sum(Error_test2_ANN,2)/sum(CV2.TestSize));
    h_opt(k) = NHiddenUnits(ind_opt_ANN);
    
    netwrkOuter = nr_main(X_train(:,2:end), y_train, X_test(:,2:end), y_test, NHiddenUnits(ind_opt_ANN));
    
    y_train_est = netwrkOuter.t_pred_train;    
    y_test_est = netwrkOuter.t_pred_test;        

    % Compute least squares error
    Error_train_ANN(k) = sum((y_train-y_train_est).^2)/CV.TrainSize(k);
    Error_test_ANN(k) = sum((y_test-y_test_est).^2)/CV.TestSize(k); 
    
    % Select optimal value of lambda
    [val,ind_opt]=min(sum(Error_test2,2)/sum(CV2.TestSize));
    lambda_opt(k)=lambda_tmp(ind_opt);  
    
    % Standardize datasets in outer fold, and save the mean and standard
    % deviations since they're part of the model (they would be needed for
    % making new predictions)
    mu(k,  :) = mean(X_train(:,2:end));
    sigma(k, :) = std(X_train(:,2:end));

    X_train_std = X_train;
    X_test_std = X_test;
    X_train_std(:,2:end) = (X_train(:,2:end) - mu(k , :)) ./ sigma(k, :);
    X_test_std(:,2:end) = (X_test(:,2:end) - mu(k, :)) ./ sigma(k, :);
        
    % Estimate w for the optimal value of lambda
    Xty=(X_train_std'*y_train);
    XtX=X_train_std'*X_train_std;
    
    regularization = lambda_opt(k) * eye(M);
    regularization(1,1) = 0; 
    w_rlr(:,k) = (XtX+regularization)\Xty;
    
    % evaluate training and test error performance for optimal selected value of
    % lambda
    Error_train_rlr(k) = sum((y_train-X_train_std*w_rlr(:,k)).^2)/CV.TrainSize(k);
    Error_test_rlr(k) = sum((y_test-X_test_std*w_rlr(:,k)).^2)/CV.TestSize(k);
    
    % Compute squared error without using the input data at all
    Error_train_nofeatures(k) = sum((y_train-mean(y_train)).^2)/CV.TrainSize(k);
    Error_test_nofeatures(k) = sum((y_test-mean(y_train)).^2)/CV.TestSize(k);
end

%%

% Computing and plotting generalization errors.

linRegEgen = sum(CV.TestSize*Error_test_rlr/N);
ANNEgen = sum(CV.TestSize*Error_test_ANN/N);
baselineEgen = sum(CV.TestSize*Error_test_nofeatures/N);

figure
bar([1,2,3],[linRegEgen,ANNEgen,baselineEgen])

%%

% Display the trained network 
mfig('Trained Network');
displayNetworkRegression(netwrkOuter);

%%

% Plotting training errors for every rlr model for each fold

figure
for i = 1:K
    % subplot(5,2,i)
    semilogx(lambda_tmp,Error_train2(:,i))
    hold on
end

%%

% Plotting training errors for every ANN model for each fold

figure
for i = 1:K
    % subplot(5,2,i)
    plot(NHiddenUnits,Error_train2_ANN(:,i))
    hold on
end

%%

% Plotting generalization errors for every ANN model for each fold

figure
for i = 1:K
    plot(NHiddenUnits,ANNEgens(:,i))
    hold on
end

%%

% Plotting generalization errors for every rlr model for each fold

figure
for i = 1:K
    semilogx(lambda_tmp,lineRegEgens(i,:))
    hold on
end

%%

figure
for k = 1:10
    plot(NHiddenUnits,Error_train2_ANN(:,k),'-o','Color','blue')
    hold on
    plot(NHiddenUnits,Error_test2_ANN(:,k),'-o','Color','red')
end
legend('Training error','Test error')

%%

figure
for k = 1:K
    semilogx(lambda_tmp,Error_train2(:,k),'-o','Color','blue')
    hold on
    semilogx(lambda_tmp,Error_test2(:,k),'-o','Color','red')
end
legend('Training error','Test error')



