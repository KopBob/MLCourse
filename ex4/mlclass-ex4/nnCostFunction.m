function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%


% [m, n] = size(X);
% X = [ones(m, 1) X];
% K = num_labels;
% yK = eye(K);

% a_1 = X;

% z_2=a_1*Theta1';
% a_2=sigmoid(z_2);

% a_2 = [ones(m, 1) a_2];
% z_2= [ones(m, 1) z_2];

% z_3=a_2*Theta2';
% a_3=sigmoid(z_3);

% h_theta=a_3;




% % % Cost Function
% % for i = 1:m
% % 	for k = 1:K
% % 		J = J + (-yK(y(i),k)*log(h_theta'(k,i))) - ((1 - yK(y(i),k)) * log(1 - h_theta'(k,i)));
% % 	end
% % end
% % J = J/m;

% % % Regularization
% % reg = 0;
% % for j = 1:hidden_layer_size,
% % 	for k = 2:(input_layer_size+1),
% % 		reg = reg + Theta1(j,k)^2;
% % 	end
% % end
% % for j = 1:num_labels,
% % 	for k = 2:(hidden_layer_size+1),
% % 		reg = reg + Theta2(j,k)^2;
% % 	end
% % end
% % reg = (lambda*reg)/(2*m);
% % J = J + reg;

% yy = eye(num_labels)(y, :);

% theta1 = nn_params(1 : hidden_layer_size * (input_layer_size + 1));
% l1 = length(theta1);

% theta2 = nn_params(l1 + 1 : end);
% l2 = length(theta2);

% J = (-1 / m) * sum(diag(yy * log(h_theta') + (1 - yy) * log(1 - h_theta'))) + (lambda / (2 * m)) * (sum(theta1(1 + hidden_layer_size : end) .^ 2) ...
%     + sum(theta2(1 + num_labels : end) .^ 2));

% % Part 2: Implement the backpropagation algorithm to compute the gradients
% %         Theta1_grad and Theta2_grad. You should return the partial derivatives of
% %         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
% %         Theta2_grad, respectively. After implementing Part 2, you can check
% %         that your implementation is correct by running checkNNGradients
% %
% %         Note: The vector y passed into the function is a vector of labels
% %               containing values from 1..K. You need to map this vector into a 
% %               binary vector of 1's and 0's to be used with the neural network
% %               cost function.
% %
% %         Hint: We recommend implementing backpropagation using a for-loop
% %               over the training examples if you are implementing it for the 
% %               first time.
% %

% b_2 = zeros(m, hidden_layer_size);
% b_3 = zeros(m, num_labels);



% for t = 1:m,
% 	% Compute Error For Output Layer.
% 	for k = 1:num_labels,
% 		b_3(t,k) = a_3(t,k) - yK(y(t),k);
% 	end

% 	for k = 1:hidden_layer_size,
% 		b_2(t,k) = b_3(t,:)*Theta2(:,k+1) .* sigmoidGradient(z_2(k));
% 	end
% end


% B_2 = b_3'*a_2;
% Theta2_grad = (1/m) * B_2;


% B_1 = b_2'*a_1;
% Theta1_grad = (1/m) * B_1;


% 






theta1 = nn_params(1 : hidden_layer_size * (input_layer_size + 1));
l1 = length(theta1);

theta2 = nn_params(l1 + 1 : end);
l2 = length(theta2);

X = [ones(m, 1) X];

a1 = X;

z2 = Theta1 * a1';
size(z2)
a2 = (sigmoid(z2))';

n = size(a2, 1);

a2 = [ones(n, 1) a2];
z3 = Theta2 * a2';
h = (sigmoid(z3));

z2 = z2';
z2 = [ones(m, 1) z2];

%yy = zeros(m, num_labels);

%for i = 1 : num_labels
 %   yy(find(y == i), i) = 1;
%end

yy = eye(num_labels)(y, :);

J = (-1 / m) * sum(diag(yy * log(h) + (1 - yy) * log(1 - h))) + (lambda / (2 * m)) * (sum(theta1(1 + hidden_layer_size : end) .^ 2) ...
    + sum(theta2(1 + num_labels : end) .^ 2));
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%


% for t = 1:m,
% 	% Compute Error For Output Layer.
% 	for k = 1:num_labels,
% 		b_3(t,k) = a_3(t,k) - yK(y(t),k);
% 	end

% 	for k = 1:hidden_layer_size,
% 		b_2(t,k) = b_3(t,:)*Theta2(:,k+1) .* sigmoidGradient(z_2(k));
% 	end
% end


delta3 = h - yy';
delta2 = Theta2' * delta3 .* (sigmoidGradient(z2))';
delta2 = delta2(2 : hidden_layer_size + 1, :);


Delta2 = delta3 * a2;
Delta1 = delta2 * a1;

Theta1_grad(:, 1) = 1 / m * Delta1(:, 1);
Theta1_grad(:, 2 : input_layer_size + 1) = 1 / m * Delta1(:, 2 : input_layer_size + 1) + lambda / m * Theta1(:, 2 : input_layer_size + 1);

Theta2_grad(:, 1) = 1 / m * Delta2(:, 1);
Theta2_grad(:, 2 : hidden_layer_size + 1) = 1 / m * Delta2(:, 2 : hidden_layer_size + 1) + lambda / m * Theta2(:, 2 : hidden_layer_size + 1);




% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
