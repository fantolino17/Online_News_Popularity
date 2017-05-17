%{
----------------------
Author: Frank Antolino
Date: May 2017
----------------------
_Description_:
Parameters: 
    w0: Weight vector, x: Data features, y: Class Labels
    numLoops: iterations, epsilon: step-size, lambda: L2 Reg Param
Returns:
    Updated version of weight vector w0, of same size 
%}

function [output] = learnLogisticWeights(w0,x,y,numLoops, epsilon, lambda)
  for k = 1:numLoops
    for i = 1:length(x)
              prob = sigmoidLikelihood(x(i,:), w0, y(i));
      for j = 1:length(x(i,:))
          w0(j) = w0(j) + (epsilon*x(i,j)*(y(i)-prob) - (w0(j)/lambda) );
      end
    end
  end
  output = w0;
end

