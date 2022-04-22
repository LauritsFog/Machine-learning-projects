
% Adding Data directory to path. 
addpath(genpath('Data'));

% Reading the csv. file into a table. 
FF_table = readtable('Alg_FFire-data.csv');

attributeNames = FF_table(:,5:14).Properties.VariableNames;

%% Principal component analysis: 

% Not considering date and class for the SVD. 
A = table2array(FF_table(:,5:14));

% Removing row with NaN's and missing classification. 
A(166,:) = [];

% Extracting the classifications. 
class = table2array(FF_table(:,15));
class(166) = [];

% Normalizing our data. 
Anormalized = (A - mean(A))./std(A);

% Computing the SVD. 
[U,S,V] = svd(Anormalized);

% Computing the explained variance.  
rho = diag(S).^2./sum(diag(S).^2);

threshold = 0.90;

%%

figure(1);
hold on
plot(rho, 'x-');
plot(cumsum(rho), 'o-');
plot([0,length(rho)], [threshold, threshold], 'k--');
legend({'Individual','Cumulative','Threshold'},'Location','best');

%%

figure(2);
gscatter(Anormalized(:,1),Anormalized(:,2),class);

%%

Z = U*S;
proj = (V'*Anormalized')';

%%

figure(3);
gscatter(Z(:,1),Z(:,2),class);

%%

% The same as figure(3). 

figure(4);
gscatter(proj(:,1),proj(:,2),class);

%%

pcs = 1:4;

figure(5);
bar(V(:,1:4));
legendCell = cellstr(num2str(pcs', 'PC%-d'));
legend(legendCell, 'location','best');
set(gca,'xticklabel',attributeNames);

%%

figure(6);
z = zeros(1,size(V,2))';
quiver(z,z,V(:,1), V(:,2), 1,'Color', 'k','AutoScale','off','LineWidth', .1);
hold on
for pc=1:10
    text(V(pc,1), V(pc,2),attributeNames{pc},'FontSize', 10)
end
xlabel('PC1')
ylabel('PC2')
grid; box off; axis equal;
% Add a unit circle
plot(cos(0:0.01:2*pi),sin(0:0.01:2*pi));
axis tight

