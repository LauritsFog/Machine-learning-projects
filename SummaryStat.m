                                %% Read the datatable
                                
%Getting the file path to where the data is located. 
cdir = fileparts(mfilename('fullpath'));
%Located in a subfolder Called Data
file_path = fullfile(cdir,'/Data/Alg_FFire-data.csv');
%Reading the data from the csv file to matlab
%The data splits into a new region at ID 123 read the Readme file in the
%Data folder for more information.
FF_table = readtable(file_path);


                %% Basic Summary statistics of the attributes

% We will focus on Temperature, Relative Humidity, wind speed, Rain and
% Fire weather index

Temp = table2array(FF_table(:,5));
RH = table2array(FF_table(:,6));
Ws = table2array(FF_table(:,7));
Rain = table2array(FF_table(:,8));
FWI = table2array(FF_table(:,14));

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

% Range = difference between max and min

%% Data as table

%Col=[Temperature, Relativ humidity, wind speed, rain, fire weather index]
%rows=[Mean,std,Median,range]
tabel_stat=[Temp_stat',RH_stat',Ws_stat',Rain_stat',FWI_stat'];

%% Correlation and similarity

X = [Temp,RH,Ws,Rain,FWI];

% Giver correlations matrix 
corr(X);

% giver plot af correlationerne.
VN = {'Temp','RH','Ws','Rain','FWI'};
corrplot(X,'varNames',VN)


% Summary:
% From the correlation coefficiants the two that are the most correlated
% are Temperature and Relativ Humidity which would be logical.

% Furthermore the Fire Weather Index also seems to be correltet a little
% with Temperature and Relativ humidity










