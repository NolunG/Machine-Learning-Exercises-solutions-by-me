function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = size(theta); %cunmber of dimensions
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%My Code For Computing Cost==================================
hx=0;
for i=1:m,
    hx=sigmoid(X(i,:)*theta);
    J = J + -y(i)*log(hx) - (1-y(i))*log(1-hx) ;
end;
for j=2:n,
    J=J + lambda * (theta(j)^2) *(1/2);
end;
J=J/m;
%Code For Gradient=========================================
Xt=X.';
for i=1:m,
    grad=grad + (sigmoid(X(i,:)*theta) - y(i))*Xt(:,i);
end;

grad = grad + (lambda)*theta;
grad(1) = grad(1) - (lambda)*theta(1);   %because we don't regulrize theta0 
grad = grad/m;                                                %,constant's multiplier

% =============================================================

end
