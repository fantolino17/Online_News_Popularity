%{
----------------------
Author: Frank Antolino
Date: May 2017
----------------------
_Description_:
Parameters: 
    x: Data Features, w: (Learned) Weight Vector
Returns:
    Vector of predicted class labels, of size x(:,1)
%}

function [output] = logisticClassify(x,w)

  %preallocate output size for better memory
  sz = length(x);
  classLabels = ones(sz);
  
  for i = 1:length(x)
    prod = 0;
    for j = 1:length(x(i,:))
      prod = prod + (x(i,j)*w(j));
    end

    if prod>=0
      classLabels(i) = 1;
    elseif prod<0
      classLabels(i) = 0;
    end
  end
  output = classLabels;
end
