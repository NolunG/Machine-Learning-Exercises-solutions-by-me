function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
Ans = Theta1*(X.');   %GETTING Z(2) for the lecture slide denotion
Ans = sigmoid(Ans);  %a(3)  
Ans = [ones(1,m); Ans];   %adding row of 1 means adding first unit in the layer
Ans = Theta2*Ans;     %GETTING Z(3)
Ans = sigmoid(Ans);   %A(3)
Ans = Ans.';
[M,I] = max(Ans, [] ,2);
p = I;

% =========================================================================


end
