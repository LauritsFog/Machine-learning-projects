%%
%Getting the file path to where the data is located. 
cdir = fileparts(mfilename('fullpath'));
%Located in a subfolder Called Data
file_path = fullfile(cdir,'/Data/Alg_FFire-data.csv');
%Reading the data from the csv file to matlab
%The data splits into a new region at ID 123 read the Readme file in the
%Data folder for more information.
FF_table = readtable(file_path);

%%
