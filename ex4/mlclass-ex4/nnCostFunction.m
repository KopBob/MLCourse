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
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


[m n] = size(X);

% Forward Propagation
X = [ones(m, 1) X];
X = X';

a1 = X;
z2 = Theta1*a1;
a2 = [ones(1, m); sigmoid(z2)];

z3 = Theta2*a2;
a3 = sigmoid(z3);
h = a3;

% Cost Function
yy = eye(num_labels)(y, :);

for i = 1:m,
	for k = 1:num_labels,
		y_ = yy(i, k);
		h_ = h(k, i);
		J = J + ( -y_*log(h_) - (1 - y_)*log(1 - h_) );
	end
end

reg1 = 0;
for i = 1:input_layer_size,
	for j = 1:hidden_layer_size,
		reg1 = reg1 + Theta1(j, i+1)^2;
	end
end


reg2 = 0;
for i = 1:hidden_layer_size,
	for j = 1:num_labels,
		reg2 = reg2 + Theta2(j, i+1)^2;
	end
end

reg = (lambda / (2 * m)) * (reg1 + reg2);

J = J/m;
J = J + reg;


% Backpropagation

% delta3 = zeros(num_labels, m);
delta2 = zeros(hidden_layer_size, m);
delta3 = h - yy';

Theta2 = Theta2(:,2:end);
Theta1 = Theta1(:,2:end);


for t = 1:m,
	% Compute Error For Output Layer.

	for k = 1:hidden_layer_size,
		delta2(k, t) = (Theta2(:,k)' * delta3(:,t)) ;
	end
end
delta2 = delta2 .* (sigmoidGradient(z2));

B2 = delta3*a2';
Theta2_grad = (1/m) * B2 .+ (lambda/m)*sum(sum(Theta2));

B1 = delta2*a1';
Theta1_grad = (1/m) * B1 .+ (lambda/m)*sum(sum(Theta2));


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
