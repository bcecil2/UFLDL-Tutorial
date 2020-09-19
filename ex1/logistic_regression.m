function [f,g] = logistic_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A column vector containing the parameter values to optimize.
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %

  m = size(X,2);
  n = size(X,1);
  
  
  
  f = 0;
  g = zeros(size(theta));
  
  sig = @(x) 1 ./ (1 + exp(-x));
  
  for i=1:m
      h = sig(theta'*X(:,i));
      f = f + y(m).*log(h) + (1-y(i)).*(log(1 - h));
  end
  f = -f;
  
  g = X*(sig(theta'*X) - y)';
end