%{
----------------------
Author: Frank Antolino
Date: May 2017
----------------------
_Description_:
Parameters:
    x: Data Features, w: Weight Vector, y: Class Labels
Returns:
    Output of sigmoid function g(h) where h = w*x
%}


function [output] = sigmoidLikelihood(x,w,y)
  sum = 0;

  for i = 1:length(x)
    sum = sum + (x(i)*w(i));
  end

  if y==1
    output = 1/(1+(exp(-sum)));
  else
    output = 1-(1/(1+exp(-sum)));
  end

end

