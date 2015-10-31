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
[m,~] = size(y);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

len = length(y);
yy = zeros(len,num_labels);
for i = 1:len
    yy(i,y(i)) = 1;
end
yy = yy;

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
a1 = X;
a1 = [ones(m, 1) a1];
z2 = a1 * Theta1';
a2 = sigmoid(z2);

mm = size(a2);
a2 = [ones(m, 1) a2];
z3 = a2 * Theta2';
a3 = sigmoid(z3);
h = a3;
temp = sum(-yy .* log(h) - (1 - yy) .* log(1 - h));
J = (1 / m) * sum(temp);


%µÚ¶þÌâ
% a1 = X;
% a1 = [ones(m, 1) a1];
% z2 = a1 * Theta1';
% a2 = sigmoid(z2);
% 
% mm = size(a2);
% a2 = [ones(mm, 1) a2];
% z3 = a2 * Theta2';
% a3 = sigmoid(z3);
% h = a3;
% 
% temp1 =(1 / m) * sum(sum(-yy .* log(h) - (1 - yy) .* log(1 - h)));
% temp3 = Theta1(:,2:end).^2;
% temp4 = Theta2(:,2:end).^2;
% temp2 = sum(sum(temp3)) + sum(sum(temp4));%%ffffffffffuuuuuuuuuuucccccccckkkkk
% J = temp1 + (lambda / (2 * m)) * temp2;

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
% DD1 = zeros(input_layer_size+1,hidden_layer_size);
% DD2 = zeros(hidden_layer_size+1,num_labels);
% delta3 = a3 - yy;
% delta2 = delta3 * Theta2(:,2:end).* sigmoidGradient(z2);
% 
% DD2 = DD2 + a2' * delta3;
% DD1 = DD1 + a1' * delta2;
% Theta1_grad = 1/m*DD1';
% Theta2_grad = 1/m*DD2';




% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

DD1 = zeros(input_layer_size+1,hidden_layer_size);
DD2 = zeros(hidden_layer_size+1,num_labels);
delta3 = a3 - yy;
delta2 = delta3 * Theta2(:,2:end).* sigmoidGradient(z2);

DD2 = DD2 + a2' * delta3;
DD1 = DD1 + a1' * delta2;
Theta1_grad = 1/m*DD1';
Theta2_grad = 1/m*DD2';

Theta1temp = [zeros(hidden_layer_size,1) Theta1(:,2:end)];
Theta2temp = [zeros(num_labels,1) Theta2(:,2:end)];

Theta1_grad = Theta1_grad + (lambda/m)*Theta1temp;
Theta2_grad = Theta2_grad + (lambda/m)*Theta2temp;











% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
