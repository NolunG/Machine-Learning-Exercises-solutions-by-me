function [mu sigma2] = estimateGaussian(X)
%ESTIMATEGAUSSIAN This function estimates the parameters of a 
%Gaussian distribution using the data in X
%   [mu sigma2] = estimateGaussian(X), 
%   The input X is the dataset with each n-dimensional data point in one row
%   The output is an n-dimensional vector mu, the mean of the data set
%   and the variances sigma^2, an n x 1 vector
% 

% Useful variables
[m, n] = size(X);

% You should return these values correctly
mu = zeros(1,n );
sigma2 = zeros(n, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the mean of the data and the variances
%               In particular, mu(i) should contain the mean of
%               the data for the i-th feature and sigma2(i)
%               should contain variance of the i-th feature.
%
mu = sum(X)/m;

summ = X;
for i = 1:m
    summ(i,:)=summ(i,:)- mu;  %subtracting corresponding mu from every feature of each data
end;
mu = mu.';  %because wme have to return acolumn vector
sigma2 = sum(summ.^2)/m;
sigma2 = sigma2.';

% =============================================================


end
