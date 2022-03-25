load_data;

addpath(genpath('Tools'));

attributeNames = FF_table(:,5:14).Properties.VariableNames;

% Extracting X and Y data. X = RH, Ws, Rain and Y = temp. 
X = table2array(FF_table(:,6:8));
Y = table2array(FF_table(:,5));

% Removing row with NaN's and missing classification. 
X(166,:) = [];
Y(166,:) = [];

%%

K1 = 10;
K2 = 10;

lambda = 10.^(-5:8);
h = [1:10];

S = length(h);

ANNValTrainTrainErr = [K1,K2];
ANNValTrainTestErr = [K1,K2];
regularizedLinRegValTrainErr = [K1,K2];
regularizedLinRegValTestErr = [K1,K2];

for i = 1:K1
    CV1 = cvpartition(length(X),'KFold',K1);
    
    xTrainOuter = X(CV1.training(i),:);
    xTestOuter = X(CV1.test(i),:);
    yTrainOuter = Y(CV1.training(i));
    yTestOuter = Y(CV1.training(i));
    
    for j = 1:K2
        
        CV2 = cvpartition(length(xTrainOuter),'KFold',K2);
        
        xTrainInner = xTrainOuter(CV2.training(j),:);
        xTestInner = xTestOuter(CV2.test(j),:);
        yTrainInner = yTrainOuter(CV2.training(j));
        yTestInner = yTrainOuter(CV2.test(j));
        
        for s = 1:S

            % Training the models with different h and lambda values.

            % Training neural network.

            ANNResults = nr_main(xTrainInner,yTrainInner,xTestInner,yTestInner,h(s));

            ANNValTrainTestErr(s,j) = ANNResults.mse_test(end);
            ANNValTrainTrainErr(s,j) = ANNResults.mse_train(end);

            % Training regularized linear regression model. 

            mu = mean(xTrainInner(:,2:end));
            sigma = std(xTrainInner(:,2:end));
            xTrainInner(:,2:end) = (xTrainInner(:,2:end) - mu) ./ sigma;
            xTestInner(:,2:end) = (xTestInner(:,2:end) - mu) ./ sigma;

            Xty = xTrainInner' * yTrainInner;
            Xtx = xTrainInner' * xTrainInner;

            regularization = lambda(s) * eye(M);
            regularization(1,1) = 0; % Remove regularization of bias-term
            w(:,s,j)=(Xtx+regularization)\Xty;
            
            % Evaluate training and test performance.
            
            regularizedLinRegValTraintErr(s,j) = sum((y_train2-X_train2*w(:,s,j)).^2);
            regularizedLinRegValTestErr(s,j) = sum((y_test2-X_test2*w(:,s,j)).^2);
        end
    end
end

%%


            
            
                         