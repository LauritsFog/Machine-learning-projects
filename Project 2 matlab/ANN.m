load_data;

addpath(genpath('Tools'));

%%

attributeNames = FF_table(:,5:14).Properties.VariableNames;

% Extracting X and Y data. X = RH, Ws, Rain and Y = temp. 
X = table2array(FF_table(:,6:8));
Y = table2array(FF_table(:,5));

% Removing row with NaN's and missing classification. 
X(166,:) = [];
Y(166,:) = [];

%%

% 2/3 for training, 1/3 for testing. 

xTrain = X(floor(length(X)/3)+1:end,:);
xTest = X(1:floor(length(X)/3),:);

yTrain = Y(floor(length(Y)/3)+1:end,:);
yTest = Y(1:floor(length(Y)/3),:);

%%

figure(1)

for N = 1:10
    
    results = nr_main(xTrain,yTrain,xTest,yTest,N);

    % Plot the error
    
    subplot(5,2,N)
    x_axis = 0:length(results.mse_test)-1;
    plot(x_axis,results.mse_test,'r*-',x_axis,results.mse_train,'bo-')
    xlabel('Number of hyperparameter updates')
    ylabel('Mean square error')
    legend('Test set','Training set')

end
    
%%

% Plot the evolution of the hyperparameters
figure(2)
subplot(2,1,1)
plot(x_axis,results.alpha,'b*-')
xlabel('Number of hyperparameter updates')
ylabel('alpha value')
subplot(2,1,2)
plot(x_axis,results.beta,'b*-')
xlabel('Number of hyperparameter updates')
ylabel('beta value')

%%

figure(3)
subplot(3,1,1)
scatter(xTest(:,1),results.t_pred_test)
subplot(3,1,2)
scatter(xTest(:,2),results.t_pred_test)
subplot(3,1,3)
scatter(xTest(:,3),results.t_pred_test)