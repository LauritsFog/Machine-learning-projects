% This is a script that shows the usage of the binary neural classifier
% when training on the pima indian data set

% Load the pima indian data:
% The data set containes 2 classes and 7 inputs. There are 200 examples used
% for training and 332 for testing.
% See 'Pattern Recognition and Neural Networks' by B.D. Ripley (1996) page 13 for detailes.
load pima_indian_data

% Set the number of hidden units
Nh = 10;

% Train the network
disp('Network training, this will not take long...')
results = nc_main(x,t,x_test,t_test,Nh);

% Plot the error 
figure(1)
x_axis = 0:length(results.Etest)-1;
plot(x_axis,results.Etest,'r*-',x_axis,results.Etrain,'bo-')
xlabel('Number of hyperparameter updates')
ylabel('Average cross-entropy error')
legend('Test set','Training set')

% Plot the classification error
figure(2)
plot(x_axis,results.Ctest,'r*-',x_axis,results.Ctrain,'bo-')
xlabel('Number of hyperparameter updates')
ylabel('Classification error')
legend('Test set','Training set')

% Plot the evolution of the hyperparameters
figure(3)
plot(x_axis,results.alpha,'b*-')
xlabel('Number of hyperparameter updates')
ylabel('alpha value')
