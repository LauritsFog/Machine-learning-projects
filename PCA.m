
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

% Subtracting the mean. 
A = A - mean(A);

% Computing the SVD. 
[U,S,V] = svd(A);

% Computing the explained variance.  
rho = diag(S).^2./sum(diag(S).^2);

threshold = 0.95;

%%

figure(1);
hold on
plot(rho, 'x-');
plot(cumsum(rho), 'o-');
plot([0,length(rho)], [threshold, threshold], 'k--');
legend({'Individual','Cumulative','Threshold'},'Location','best');

%%

figure(2);
gscatter(A(:,1),A(:,2),class);

%%

Z = U*S;

%%

figure(3);
gscatter(Z(:,1),Z(:,2),class);

%%

pcs = 1:2;

figure(4);
bar(V(:,1:2));
legendCell = cellstr(num2str(pcs', 'PC%-d'));
legend(legendCell, 'location','best');
set(gca,'xticklabel',attributeNames);

%%

figure(5);
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

