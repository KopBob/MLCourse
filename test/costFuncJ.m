%% costFuncJ: Compute cost function
function J = costFuncJ(X, y, theta)

m  = size(X,1)
prediction = X*theta
% examples
sqrErrors = (prediction-y).^2


J = 1/(2*m) * sum(sqrErrors);