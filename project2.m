%% Load data
load_data;
A_Names=FF_table.Properties.VariableNames;
arr_FF_table=table2array(FF_table(:,1:14));
%% Regression, part a

%% 1
%Regression of temperature using humidity, rain, wind

y = arr_FF_table(:, 5);%Temp
Xr = arr_FF_table(:, [6,7,8]);%Humidity, rain, wind

w_est = glmfit(Xr, y, 'normal');
y_est = glmval(w_est, Xr, 'identity');
%Scatter plot of predicted vs actual
figure(1)
plot(y, y_est, '.');
xlabel('Temp (true)');
ylabel('Temp (estimated)');

mfig('Residual error');
histogram(y-y_est, 40);

%% Normalized Xr
Xr_norm=normalize(Xr);%(Xr - mean(Xr))./std(Xr);
X=Xr_norm;
w_est = glmfit(X, y, 'normal');

y_est = glmval(w_est, X, 'identity');
%Scatter plot of predicted vs actual
figure(2)
plot(y, y_est, '.');
xlabel('Temp (true)');
ylabel('Temp (estimated)');












%% Regularized Linear regression 
% include an additional attribute corresponding to the offset
X=[ones(size(Xr_norm,1),1) Xr_norm];
M=3+1;
attributeNames={'Offset', A_Names{6:8}};
 
% Crossvalidation
% Create crossvalidation partition for evaluation of performance of optimal
% model
K = 10;
N=244;
CV = cvpartition(N, 'Kfold', K);

% Values of lambda
lambda_tmp=10.^(-5:8);

% Initialize variables
T=length(lambda_tmp);
Error_train = nan(K,1);
Error_test = nan(K,1);
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
        X_train2(:,2:end) = (X_train2(:,2:end) - mu2) ./ sigma2;
        X_test2(:,2:end) = (X_test2(:,2:end) - mu2) ./ sigma2;
        
        Xty2 = X_train2' * y_train2;
        XtX2 = X_train2' * X_train2;
        for t=1:length(lambda_tmp)   
            % Learn parameter for current value of lambda for the given
            % inner CV_fold
            regularization = lambda_tmp(t) * eye(M);
            regularization(1,1) = 0; % Remove regularization of bias-term
            w(:,t,kk)=(XtX2+regularization)\Xty2;
            % Evaluate training and test performance
            Error_train2(t,kk) = sum((y_train2-X_train2*w(:,t,kk)).^2);
            Error_test2(t,kk) = sum((y_test2-X_test2*w(:,t,kk)).^2);
        end
    end    
    
    % Select optimal value of lambda
    [val,ind_opt]=min(sum(Error_test2,2)/sum(CV2.TestSize));
    lambda_opt(k)=lambda_tmp(ind_opt);    

    % Display result for last cross-validation fold (remove if statement to
    % show all folds)
    if k == K
        mfig(sprintf('(%d) Regularized Solution',k));    
        subplot(1,2,1); % Plot error criterion
        semilogx(lambda_tmp, mean(w(2:end,:,:),3),'.-');
        % For a more tidy plot, we omit the attribute names, but you can
        % inspect them using:
        %legend(attributeNames(2:end), 'location', 'best');
        xlabel('\lambda');
        ylabel('Coefficient Values');
        title('Values of w');
        subplot(1,2,2); % Plot error        
        loglog(lambda_tmp,[sum(Error_train2,2)/sum(CV2.TrainSize) sum(Error_test2,2)/sum(CV2.TestSize)],'.-');   
        legend({'Training Error as function of lambda','Test Error as function of lambda'},'Location','SouthEast');
        title(['Optimal value of lambda: 1e' num2str(log10(lambda_opt(k)))]);
        xlabel('\lambda');           
        drawnow;    
    end
    
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
    Error_train_rlr(k) = sum((y_train-X_train_std*w_rlr(:,k)).^2);
    Error_test_rlr(k) = sum((y_test-X_test_std*w_rlr(:,k)).^2);
    
    % Compute squared error without regularization
    w_noreg(:,k)=XtX\Xty;
    Error_train(k) = sum((y_train-X_train_std*w_noreg(:,k)).^2);
    Error_test(k) = sum((y_test-X_test_std*w_noreg(:,k)).^2);
    
    % Compute squared error without using the input data at all
    Error_train_nofeatures(k) = sum((y_train-mean(y_train)).^2);
    Error_test_nofeatures(k) = sum((y_test-mean(y_train)).^2);
     
end


fprintf('\n');
fprintf('Linear regression without feature selection:\n');
fprintf('- Training error: %8.2f\n', sum(Error_train)/sum(CV.TrainSize));
fprintf('- Test error:     %8.2f\n', sum(Error_test)/sum(CV.TestSize));
fprintf('- R^2 train:     %8.2f\n', (sum(Error_train_nofeatures)-sum(Error_train))/sum(Error_train_nofeatures));
fprintf('- R^2 test:     %8.2f\n', (sum(Error_test_nofeatures)-sum(Error_test))/sum(Error_test_nofeatures));
fprintf('Regularized linear regression:\n');
fprintf('- Training error: %8.2f\n', sum(Error_train_rlr)/sum(CV.TrainSize));
fprintf('- Test error:     %8.2f\n', sum(Error_test_rlr)/sum(CV.TestSize));
fprintf('- R^2 train:     %8.2f\n', (sum(Error_train_nofeatures)-sum(Error_train_rlr))/sum(Error_train_nofeatures));
fprintf('- R^2 test:     %8.2f\n', (sum(Error_test_nofeatures)-sum(Error_test_rlr))/sum(Error_test_nofeatures));

fprintf('\n');
fprintf('Weight in last fold: \n');
for m = 1:M
    disp( sprintf(['\t', attributeNames{m},':\t ', num2str(w_rlr(m,end))]))
end
%disp(w_rlr(:,end))





































