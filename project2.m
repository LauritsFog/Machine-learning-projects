%% Load data
load_data;

%% Regression, part a

%% 1
%Trying to predict FWI from the normal weather observations
arr_FF_table=table2array(FF_table(:,1:14));

y = arr_FF_table(:, 14);
Xr = arr_FF_table(:, 5:8);

w_est = glmfit(Xr, y, 'normal');

y_est = glmval(w_est, Xr, 'identity');
%Scatter plot of predicted vs actual
figure(1)
plot(y, y_est, '.');
xlabel('FWI (true)');
ylabel('FWI (estimated)');

mfig('Residual error');
histogram(y-y_est, 40);

%% Normalized Xr
Xr_norm=normalize(Xr);%(Xr - mean(Xr))./std(Xr);
w_est = glmfit(Xr_norm, y, 'normal');

y_est = glmval(w_est, Xr_norm, 'identity');
%Scatter plot of predicted vs actual
figure(2)
plot(y, y_est, '.');
xlabel('FWI (true)');
ylabel('FWI (estimated)');


