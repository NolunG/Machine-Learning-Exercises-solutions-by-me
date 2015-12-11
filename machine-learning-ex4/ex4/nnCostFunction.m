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

% Add ones to the X data matrix
X = [ones(m, 1) X];

%Make matrix Y from vector y with bianries
Y = zeros(m,num_labels);
for i=1:m,
    Y(i,y(i))=1;
end;

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
%FeedForward===========================================================

z2=X*(Theta1.');%5000*25...for this example
A2 = sigmoid(z2);
A2 = [ones(m,1) A2];  %Added one colun so that each a can get added 1 
z3 = A2*(Theta2.');   %I did transpose calculations in previous exercise
A3 = sigmoid(z3);

J = sum(sum(-Y.*log(A3)-(1-Y).*log(1-A3)));
%====for regularization======================================
theta1 = Theta1;
theta2 = Theta2;

theta1(:,1) = 0;
theta2(:,1) = 0;
%theta1(1,:) = 0;
%theta2(1,:) = 0;

J = J + (lambda/2)*(sum(sum(theta1.^2))+sum(sum(theta2.^2)));
J = J/m;

%==========BackPropogation========================================
for t=1:m,
    
Del3 = (A3(t,:)-Y(t,:)).';  % cause del3 is a column vector

Del2 = ((Theta2.')*Del3).*[1;sigmoidGradient(z2(t,:).')];%caution here

Del2 = Del2(2:end);

Theta1_grad = Theta1_grad + Del2*(X(t,:));  

Theta2_grad = Theta2_grad + Del3*(A2(t,:));

end;

Theta1_grad = Theta1_grad + lambda*(theta1);
Theta2_grad = Theta2_grad + lambda*(theta2);

Theta1_grad = Theta1_grad/m;
Theta2_grad = Theta2_grad/m;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
