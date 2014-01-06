function plotDataPoints(X, idx, K)
%PLOTDATAPOINTS plots data points in X, coloring them so that those with the same
%index assignments in idx have the same color
%   PLOTDATAPOINTS(X, idx, K) plots data points in X, coloring them so that those 
%   with the same index assignments in idx have the same color

% % Create palette
% palette = hsv(K + 1);
% colors = palette(idx, :);

% % Plot the data
% scatter(X(:,1), X(:,2), 15, colors);

subset = randperm(size(X, 1))(1:99);
Xsubset = X(subset, :);
idxsubset = idx(subset);

% Create palette
palette = hsv(K + 1);
colors = palette(idxsubset, :);

% Plot the data

scatter(Xsubset(:,1), Xsubset(:,2), 15, colors);

end
