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
corr(X);

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

% Skiller ved ID 123
% Fra 1 - 122  Bejaia
% Fra 123 til 244  Sidi Bel-abbes


x = [1:244];

figure(3)
title('Plot af Data')

subplot(3,3,1)
hold on
scatter(x(1:122),Temp(1:122),'blue')
scatter(x(1:122),Temp(123:244),'red')
hold off
title('Temperature')

subplot(3,3,2)
hold on
scatter(x(1:122),RH(1:122),'blue')
scatter(x(1:122),RH(123:244),'red')
hold off
title('Relative Humidity')

subplot(3,3,3)
hold on
scatter(x(1:122),Ws(1:122),'blue')
scatter(x(1:122),Ws(123:244),'red')
hold off
title('Wind speed')

subplot(3,3,4)
hold on
scatter(x(1:122),Rain(1:122),'blue')
scatter(x(1:122),Rain(123:244),'red')
hold off
title('Rain')

subplot(3,3,5)
hold on
scatter(x(1:122),FWI(1:122),'blue')
scatter(x(1:122),FWI(123:244),'red')
hold off
title('Fire Weather Index')

subplot(3,3,6)
hold on
scatter(x(1:122),FFMC(1:122),'blue')
scatter(x(1:122),FFMC(123:244),'red')
hold off
title('Fine Fuel Moisture Code index')

subplot(3,3,7)
hold on
scatter(x(1:122),DC(1:122),'blue')
scatter(x(1:122),DC(123:244),'red')
hold off
title('Drought Code index')

% Construct a Legend with the data from the sub-plots
hL = legend({'Bejaia Region', 'Sidi Bel-abbes Region'},'FontSize',14);
% Programatically move the Legend
newPosition = [0.5 0.1 0.3 0.1];
newUnits = 'normalized';
set(hL,'Position', newPosition,'Units', newUnits);


%% Correlation for all attributes

Xfull = table2array(FF_table(:,5:14));
VN = {'Temp','RH','Ws','Rain','FFMC','DMC','DC','ISI','BUI','FWI'};
corrplot(Xfull,'varNames',VN)

