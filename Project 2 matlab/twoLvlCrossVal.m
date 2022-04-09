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

% Initializing for RLR
lambda_tmp=[0:14];

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
Error_train_gen_rlr = zeros(K,T);
Error_test_gen_rlr = zeros(K,T);

% Initializing for ANN

NHiddenUnits = [1:10];

Error_train_ANN = nan(K,1);
Error_test_ANN = nan(K,1);
Error_train2_ANN = nan(length(NHiddenUnits),K);
Error_test2_ANN = nan(length(NHiddenUnits),K);
bestnet=cell(K,1);
h_opt = nan(K,1);
Error_train_gen_ANN = zeros(K,length(NHiddenUnits));
Error_test_gen_ANN = zeros(K,length(NHiddenUnits));

ANN_test_est = cell(K,1);
RLR_test_est = cell(K,1);
Baseline_test_est = cell(K,1);
y_true = cell(K,1);

% For each crossvalidation fold
for k = 1:K
    fprintf('Crossvalidation fold %d/%d\n', k, K);
    
    % Extract the training and test set
    X_train = X(CV.training(k), :);
    y_train = y(CV.training(k));
    X_test = X(CV.test(k), :);
    y_test = y(CV.test(k));
    y_true{k,1} = y_test;

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
            Error_train2(t,kk) = sum((y_train2-X_train2_std*w(:,t,kk)).^2);
            Error_test2(t,kk) = sum((y_test2-X_test2_std*w(:,t,kk)).^2);
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
            Error_train2_ANN(t,kk) = sum((y_train2-y_train_est).^2);
            Error_test2_ANN(t,kk) = sum((y_test2-y_test_est).^2); 
        end
    end
    
    Error_test_gen_ANN(k,:) = sum(Error_test2_ANN,2)/sum(CV2.TestSize);
    Error_train_gen_ANN(k,:) = sum(Error_train2_ANN,2)/sum(CV2.TrainSize);
    
    Error_test_gen_rlr(k,:) = sum(Error_test2,2)/sum(CV2.TestSize);
    Error_train_gen_rlr(k,:) = sum(Error_train2,2)/sum(CV2.TrainSize);
    
    % Training and testing 
    
    [val_ANN,ind_opt_ANN]=min(sum(Error_test2_ANN,2)/sum(CV2.TestSize));
    h_opt(k) = NHiddenUnits(ind_opt_ANN);
    
    netwrkOuter = nr_main(X_train(:,2:end), y_train, X_test(:,2:end), y_test, NHiddenUnits(ind_opt_ANN));
    
    y_train_est = netwrkOuter.t_pred_train;    
    y_test_est = netwrkOuter.t_pred_test;
    ANN_test_est{k,1} = y_test_est;

    % Compute least squares error
    Error_train_ANN(k) = sum((y_train-y_train_est).^2)/CV.TrainSize(k);
    Error_test_ANN(k) = sum((y_test-y_test_est).^2)/CV.TestSize(k); 
    
    % Select optimal value of lambda
    [val,ind_opt]=min(sum(Error_test2,2)/sum(CV2.TestSize));
    lambda_opt(k)=lambda_tmp(ind_opt);  
    
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
    RLR_test_est{k,1} = X_test_std*w_rlr(:,k);
    
    % Compute squared error without using the input data at all
    Error_train_nofeatures(k) = sum((y_train-mean(y_train)).^2)/CV.TrainSize(k);
    Error_test_nofeatures(k) = sum((y_test-mean(y_train)).^2)/CV.TestSize(k);
    Baseline_test_est{k,1} = ones(sum(CV.test(k)),1)*mean(y_train);
end

%%

% Computing and plotting generalization errors.

linRegEgen = sum(CV.TestSize*Error_test_rlr/N);
ANNEgen = sum(CV.TestSize*Error_test_ANN/N);
baselineEgen = sum(CV.TestSize*Error_test_nofeatures/N);

%%

% Significance value
alpha = 0.05;

% Comparing ANN with RLR

zA = (cell2mat(y_true)-cell2mat(ANN_test_est)).^2;
zB = (cell2mat(y_true)-cell2mat(RLR_test_est)).^2;

z_hat_1 = (1/N)*sum(zA-zB);

[~, p_ANN_RLR, CI_ANN_RLR] = ttest(zA-zB, [], 'alpha', alpha);

[~, ~, CIANN] = ttest(zA, [], 'alpha', alpha);
[~, ~, CIRLR] = ttest(zB, [], 'alpha', alpha);

% Comparing ANN with baseline

zA = (cell2mat(y_true)-cell2mat(ANN_test_est)).^2;
zB = (cell2mat(y_true)-cell2mat(Baseline_test_est)).^2;

z_hat_2 = (1/N)*sum(zA-zB);

[~, p_ANN_Baseline, CI_ANN_Baseline] = ttest(zA-zB, [], 'alpha', alpha);

[~, ~, CIBaseline] = ttest(zB, [], 'alpha', alpha);

% Comparing RLR with baseline

zA = (cell2mat(y_true)-cell2mat(RLR_test_est)).^2;
zB = (cell2mat(y_true)-cell2mat(Baseline_test_est)).^2;

z_hat_3 = (1/N)*sum(zA-zB);

[~, p_RLR_Baseline, CI_RLR_Baseline] = ttest(zA-zB, [], 'alpha', alpha);

%%

figure
p = plot(1,linRegEgen,'o',1,CIRLR(1),'*',1,CIRLR(2),'*',2,ANNEgen,'o',2,CIANN(1),'*',2,CIANN(2),'*',3,baselineEgen,'o',3,CIBaseline(1),'*',3,CIBaseline(2),'*');

p(1).MarkerSize = 10;
p(4).MarkerSize = 10;
p(7).MarkerSize = 10;

p(1).MarkerFaceColor = 'b';
p(4).MarkerFaceColor = 'b';
p(7).MarkerFaceColor = 'b';

p(1).Color = 'b';
p(4).Color = 'b';
p(7).Color = 'b';

p(2).Color = 'r';
p(3).Color = 'r';
p(5).Color = 'r';
p(6).Color = 'r';
p(8).Color = 'r';
p(9).Color = 'r';

p(2).MarkerSize = 10;
p(3).MarkerSize = 10;
p(5).MarkerSize = 10;
p(6).MarkerSize = 10;
p(8).MarkerSize = 10;
p(9).MarkerSize = 10;

grid on

set(gca, 'XTickLabel',{' ','RLR',' ','ANN',' ','Baseline',' '},'Fontsize',15)

legend('Generelization error', 'Confidence interval')

xlim([0.5 3.5])

%%

c = cool(3);

figure
for i = 10:10
    plot(y(CV.test(i)),'-o','Color','r')
    hold on
    plot(RLR_test_est{i},'-o','Color',c(1,:))
    hold on
    plot(ANN_test_est{i},'-o','Color',c(2,:))
    hold on
    plot(Baseline_test_est{i},'-o','Color',c(3,:))
    hold on
end
legend('True values', 'RLR predictions', 'ANN predictions','Baseline predictions','Location','northwest')

%%

zhats = [z_hat_1;z_hat_2;z_hat_3];
P = [p_ANN_RLR;p_ANN_Baseline;p_RLR_Baseline];
CI_lower = [CI_ANN_RLR(1);CI_ANN_Baseline(1);CI_RLR_Baseline(1)];
CI_upper = [CI_ANN_RLR(2);CI_ANN_Baseline(2);CI_RLR_Baseline(2)];

table(zhats,CI_lower,CI_upper,P)

ttest = [zhats CI_lower CI_upper P];

%%

figure
bar([linRegEgen,ANNEgen,baselineEgen])
set(gca, 'XTickLabel',{'RLR','ANN','Baseline'},'Fontsize',15)
title('Generalization errors')
grid on

%%

% Display the trained network 
mfig('Trained Network');
displayNetworkRegression(netwrkOuter);

%%

figure
plot([1:10],Error_train_rlr,'-o','Color','b')
hold on
plot([1:10],Error_test_rlr,'-o','Color','r')
legend('Training errors','Testing errors','Fontsize',15)
title('Errors with optimal lambda','Fontsize',15)
grid on
hold off

%%

figure
plot([1:10],Error_train_ANN,'-o','Color','b')
hold on
plot([1:10],Error_test_ANN,'-o','Color','r')
legend('Training errors','Testing errors','Fontsize',15)
title('Errors with optimal h','Fontsize',15)
grid on
hold off

%%

c = cool(10);

figure
for k = 1:1
    subplot(2,1,1)
    plot(NHiddenUnits,Error_test_gen_ANN(k,:),'-o','Color',c(k,:))
    title('Testing errors','Fontsize',15)
    grid on
    hold on
    subplot(2,1,2)
    plot(NHiddenUnits,Error_train_gen_ANN(k,:),'-o','Color',c(k,:))
    title('Training errors','Fontsize',15)
    grid on
    hold on
end
hold off

%%

c = cool(14);

figure
for k = 1:10
    subplot(2,1,1)
    plot(lambda_tmp,Error_test_gen_rlr(k,:),'-o','Color',c(k,:))
    xlim([0 lambda_tmp(end)])
    title('Testing errors','Fontsize',15)
    grid on
    hold on
    subplot(2,1,2)
    plot(lambda_tmp,Error_train_gen_rlr(k,:),'-o','Color',c(k,:))
    xlim([0 lambda_tmp(end)])
    title('Training errors','Fontsize',15)
    grid on
    hold on
end
hold off

%%

figure
plot(lambda_tmp,Error_test_gen_rlr(k,:),'-o','Color','r')
hold on
plot(lambda_tmp,Error_train_gen_rlr(k,:),'-o','Color','b')
legend
legend('Testing errors','Training errors','Fontsize',15)

