load_data;

addpath(genpath('../Tools/nr'));

attributeNames = FF_table(:,5:14).Properties.VariableNames;

% Extracting X and Y data. X = RH, Ws, Rain and Y = temp. 
X = normalize(table2array(FF_table(:,6:8)));
Y = normalize(table2array(FF_table(:,5)));

Xmod = [ones(size(X,1),1) X];

% Removing row with NaN's and missing classification. 
X(166,:) = [];
Y(166,:) = [];

Xmod(166,:) = [];

%%

K1 = 10;
K2 = 10;

basemodelMean = zeros(K1,K2);
lambda = 10.^(-4:5);
hiddenLayers = [1:10];
M = 3+1;
N = length(Y);

S = length(hiddenLayers);

ANNEtrain = zeros(K1,K2);
ANNEval = zeros(K1,K2);
linRegEtrain = zeros(K1,K2);
linRegEval = zeros(K1,K2);
baseModelEtrain = zeros(K1,1);
baseModelEval = zeros(K1,1);

baselineEgens = zeros(K1,1);
ANNEgens = zeros(K1,1);
lineRegEgens = zeros(K1,1);

baselineEtest = zeros(K1,1);
linRegEtest = zeros(K1,1);
ANNEtest = zeros(K1,1);

for i = 1:K1
    
    CV1 = cvpartition(length(X),'KFold',K1);
    
    xDpar = X(CV1.training(i),:);
    xDtest = X(CV1.test(i),:);
    yDpar = Y(CV1.training(i));
    yDtest = Y(CV1.test(i));
    
    xDparmod = X(CV1.training(i),:);
    xDtestmod = X(CV1.test(i),:);
    
    for j = 1:K2
        
        % Partioning training data from outer loop. 
        
        CV2 = cvpartition(length(xDpar),'KFold',K2);
        
        xDtrain = xDpar(CV2.training(j),:);
        xDval = xDpar(CV2.test(j),:);
        yDtrain = yDpar(CV2.training(j));
        yDval = yDpar(CV2.test(j));
        
        xDtrainmod = xDparmod(CV2.training(j),:);
        xDvalmod = xDparmod(CV2.test(j),:);
        
        for s = 1:S

            % Training the models with different hiddenLayers and lambda values.
            
            % Training neural network.

            ANNResultsInner = nr_main(xDtrain,yDtrain,xDval,yDval,hiddenLayers(s));

            % Evaluate training and test performance.
            
            ANNEval(s,j) = ANNResultsInner.mse_test(end);
            ANNEtrain(s,j) = ANNResultsInner.mse_train(end);

            % Training regularized linear regression model. 

            mu = mean(xDtrainmod(:,2:end));
            sigma = std(xDtrainmod(:,2:end));
            
            xDtrainmodstd = xDtrainmod;
            xDvalmodstd = xDvalmod;

            xDtrainmodstd(:,2:end) = (xDtrainmod(:,2:end) - mu) ./ sigma;
            xDvalmodstd(:,2:end) = (xDvalmod(:,2:end) - mu) ./ sigma;

            Xty = xDtrainmodstd' * yDtrain;
            Xtx = xDtrainmodstd' * xDtrainmodstd;

            regularization = lambda(s) * eye(M);
            regularization(1,1) = 0; % Remove regularization of bias-term
            wInner(:,s,j)=(Xtx+regularization)\Xty;
            
            % Evaluate training and test performance.
            
            linRegEtrain(s,j) = mean((yDtrain-xDtrainmod*wInner(:,s,j)).^2)*length(yDval)/length(yDpar);
            linRegEval(s,j) = mean((yDval-xDvalmod*wInner(:,s,j)).^2)*length(yDval)/length(yDpar);
        end
    end
    
    baseModelEtrain(i) = sum((yDpar-mean(yDpar)).^2);
    baseModelEval(i) = sum((yDtest-mean(yDpar)).^2);
    
    for s = 1:K2
        lineRegEgens(s) = sum(linRegEval(s,:));
        ANNEgens(s) = sum(ANNEval(s,:));
    end
    
    % Finding model with best generalization error.
    
    % [baseLineBest, baselineIdx] = min(baselineEgens);
    [linRegBest, linRegIdx] = min(lineRegEgens);
    [ANNBest, ANNIdx] = min(ANNEgens);
    
    % Training best models on outer test set. 
    
    ANNResultsOuter = nr_main(xDpar,yDpar,xDtest,yDtest,hiddenLayers(ANNIdx));
    
    mu = mean(xDparmod(:,2:end));
    sigma = std(xDparmod(:,2:end));
    xDparmod(:,2:end) = (xDparmod(:,2:end) - mu) ./ sigma;
    xDtestmod(:,2:end) = (xDtestmod(:,2:end) - mu) ./ sigma;

    Xty = xDparmod' * yDpar;
    Xtx = xDparmod' * xDparmod;

    regularization = lambda(linRegIdx) * eye(M);
    regularization(1,1) = 0; % Remove regularization of bias-term
    wOuter=(Xtx+regularization)\Xty;
    
    % Testing the best models on the outer test set. 
    
    % baselineEtest(i) = sum((yDtest-mean(basemodelMean(baselineIdx,:))).^2)/length(yDtest);
    linRegEtest(i) = mean((yDtest-xDtestmod*wOuter).^2);
    ANNEtest(i) = ANNResultsOuter.mse_test(end);
end

% baselineEgen = sum(baselineEtest*CV1.Testsize/N);
linRegEgen = sum(linRegEtest'.*CV1.TestSize/N);
ANNEgen = sum(ANNEtest'.*CV1.TestSize/N);

%%

figure
subplot(1,2,1)
plot([1:10],linRegEtest)
title('Regularized linear regression')
subplot(1,2,2)
plot([1:10],ANNEtest)
title('ANN')

figure
bar([1,2],[linRegEgen,ANNEgen])

%%

figure
for k = 1:10
    subplot(5,2,k)
    plot([1:10],ANNEtrain(:,k))
    hold on
    plot([1:10],ANNEval(:,k))
    legend('Training error','Test error')
end

%%

figure
for k = 1:10
    subplot(5,2,k)
    plot([1:10],linRegEtrain(:,k))
    hold on
    plot([1:10],linRegEval(:,k))
    legend('Training error','Test error')
end
