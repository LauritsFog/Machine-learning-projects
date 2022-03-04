                                %% Read the datatable
                                
%Getting the file path to where the data is located. 
cdir = fileparts(mfilename('fullpath'));
%Located in a subfolder Called Data
file_path = fullfile(cdir,'/Data/Alg_FFire-data.csv');
%Reading the data from the csv file to matlab
%The data splits into a new region at ID 123 read the Readme file in the
%Data folder for more information.
FF_table = readtable(file_path);

% For at skifte til den korrekte path
% cd '/Users/frederiknagel/Desktop/GitHub/GitHub'
% Virker nok kun på min computer



                %% Basic Summary statistics of the attributes

% We will focus on Temperature, Relative Humidity, wind speed, Rain and
% Fire weather index

Temp = table2array(FF_table(:,5));
RH = table2array(FF_table(:,6));
Ws = table2array(FF_table(:,7));
Rain = table2array(FF_table(:,8));
FWI = table2array(FF_table(:,14));
FFMC = table2array(FF_table(:,9));
DC = table2array(FF_table(:,11));

FWI(166) = 10.4;

%% Mean, Standart deviation, Median and Range

% We compute the mean, standart deviation, median and range for all five
% attributes in focus, and insert dem in a row vector

% row = [mean, std, median, range]

Temp_stat = [mean(Temp),std(Temp),median(Temp),range(Temp)];
RH_stat = [mean(RH),std(RH),median(RH),range(RH)];
Ws_stat = [mean(Ws),std(Ws),median(Ws),range(Ws)];
Rain_stat = [mean(Rain),std(Rain),median(Rain),range(Rain)];
FWI_stat = [mean(FWI),std(FWI),median(FWI),range(FWI)];
FFMC_stat = [mean(FFMC),std(FFMC),median(FFMC),range(FFMC)];
DC_stat = [mean(DC),std(DC),median(DC),range(DC)];

% Range = difference between max and min

%% Data as table

%Col=[Temperature, Relativ humidity, wind speed, rain, fire weather index]
%rows=[Mean,std,Median,range]
tabel_stat=[Temp_stat',RH_stat',Ws_stat',Rain_stat',FWI_stat',FFMC_stat',DC_stat'];


%% Correlation and similarity

X = [Temp,RH,Ws,Rain,FWI,FFMC,DC];

% Giver correlations matrix 
corr(X)

% giver plot af correlationerne.
figure(1)
VN = {'Temp','RH','Ws','Rain','FWI','FFMC','DC'};
corrplot(X,'varNames',VN)


% Summary:
% From the correlation coefficiants the two that are the most correlated
% are Temperature and Relativ Humidity which would be logical.

% Furthermore the Fire Weather Index also seems to be correltet a little
% with Temperature and Relativ humidity


%% Boxplot 

figure(2)
boxplot(X)


%% Scatter

x = [1:244];

figure(3)
subplot(3,3,1)
scatter(x,Temp)
title('Temperature')

subplot(3,3,2)
scatter(x,RH)
title('Relative Humidity')

subplot(3,3,3)
scatter(x,Ws)
title('Wind speed')

subplot(3,3,4)
scatter(x,Rain)
title('Rain')

subplot(3,3,5)
scatter(x,FWI)
title('Fire Weather Index')

subplot(3,3,6)
scatter(x,FFMC)
title('Fine Fuel Moisture Code index')

subplot(3,3,7)
scatter(x,DC)
title('Drought Code index')










