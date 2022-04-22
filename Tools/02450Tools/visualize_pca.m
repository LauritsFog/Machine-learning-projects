function visualize_pca(X, y, pcs, att, attributeNames, classNames)
%VISUALIZE_PCA Visualize a pair of chosen attributes and a pair of
%principal components from a PCA analysis.
%   The function takes as input a data matrix X and a grouping variable y.
%   The input X is standardized, and a PCA is done on X. Following this,
%   the function makes two plots. The first plot is a plot of a chosen pair
%   of attributes where a chosen pair of principal directions are
%   visualized as vectors (in red). The second plot is the projection of
%   all the data onto the two chosen principal directions, and in that
%   space the coefficients in the principal direction for the chosen
%   attributes are shown as vectors (in blue).
%
% Usage:
%   figure();
%   visualize_pca(X, y, pcs, att, attributeNames, classNames);
%
% Input:
%   X               N x M data matrix
%   y               N x 1 vector grouping variable (class labels)
%   pcs             2 x 1 vector of chosen principal components to show
%   att             2 x 1 vector of chosen attributes to show
%   attributeNames  M x 1 cell array of names of each of the M attributes
%   classNames      C x 1 cell array of names of each of the C classes in y
%
% Output: 
%   Two figures in a subplot visualizing the chosen attributes and
%   principal directions.
%
% Copyright 2019, Technical University of Denmark

% Standardize the data
X = zscore(X);

% Obtain the PCA solution by calculate the SVD of input X
[U, S, V] = svd(X,'econ');

% Determien projection:
Z = U*S;

% Set fontsize of vector labels
fz = 20; 

% First plot of attribute space:
subplot(1,2,1)
    % Plot scatter plot of chosen attributes
    C = length(classNames);
    hold on
    colors = get(gca,'colororder');
    for c = 0:C-1
        scatter(X(y==c,att(1)), X(y==c,att(2)), 50, 'o', ...
                'MarkerFaceColor', colors(c+1,:), ...
                'MarkerEdgeAlpha', 0, ...
                'MarkerFaceAlpha', .5);
    end
    xlabel(attributeNames{att(1)}, 'Color', 'b');
    ylabel(attributeNames{att(2)}, 'Color', 'b');
    axis equal
    
    % Plot principal directions in the attribute space, both as the unit
    % vector and as a line extending along the vector.
    % Start with the line:
    xlimits = get(gca,'XLim');
    ylimits = get(gca,'YLim');
    for k = pcs
        m = V(att(2),k)/V(att(1),k);
        y1 = m*xlimits(1);
        y2 = m*xlimits(2);
        line([xlimits(1) xlimits(2)],[y1 y2], ...
                'LineStyle','--', ...
                'Color','k',...
                'LineWidth',.01)
        text(2*V(att(1),k), 2*V(att(2),k), ...
                sprintf('PC%d', k), ...
                'Color','r', ...
                'FontSize',fz);
    end
    % The the arrow:
    quiver([0, 0],[0, 0],V(att(1),pcs),V(att(2),pcs), 1, ...
           'AutoScale','off', ...
           'Color', 'r', ...
           'MaxHeadSize', .5, ...
           'LineWidth', 1)
    xlim(xlimits);
    ylim(ylimits);
    title('Standardized attribute space');
    grid
    
% Second plot of the principal component space
subplot(1,2,2)
    % Plot projection
    C = length(classNames);
    hold on
    colors = get(gca,'colororder');
    for c = 0:C-1
        scatter(Z(y==c,pcs(1)), Z(y==c,pcs(2)), 50, 'o', ...
                'MarkerFaceColor', colors(c+1,:), ...
                'MarkerEdgeAlpha', 0, ...
                'MarkerFaceAlpha', .5);
    end
    xlabel(sprintf('PC %d', pcs(1)),'Color','r');
    ylabel(sprintf('PC %d', pcs(2)),'Color','r');
    axis equal
    
    % Determine limits to make the extended line of the attribute
    % direction:
    xlimits = get(gca,'XLim');
    ylimits = get(gca,'YLim');
    
    % Plot the attribute coefficients as a vector in the PC space as well
    % as a dashed line that spans the plot.
    % Start with the line:
    for a=att
        m = V(a,pcs(2))/V(a,pcs(1));
        y1 = m*xlimits(1);
        y2 = m*xlimits(2);
        line([xlimits(1) xlimits(2)],[y1 y2], ...
                'LineStyle','--', ...
                'Color','k',...
                'LineWidth',.01)
        text(2*V(a,pcs(1)), 2*V(a,pcs(2)),attributeNames{a}, ...
             'FontSize', fz, ...
             'Color','b')
    end
    % Then the arrow:
    z = zeros(length(att),1);
    quiver(z,z,V(att,pcs(1)), V(att,pcs(2)), 1, ...
           'AutoScale','off', ...
            'Color', 'b', ...
            'MaxHeadSize', .5, ...
           'LineWidth', 1)
    grid
    box off
    xlim(xlimits);
    ylim(ylimits);
    legend(classNames,'Location','best');
    title('Principal component space');
        
end

