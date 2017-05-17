%{
----------------------
Author: Frank Antolino
Date: May 2017
----------------------
_Description_:
Parameters: 
    x: Data Features, w: (Learned) Weight Vector, y: Actual class labels
Returns:
    Percent of x thats was correctly classified using w.
%}


function [output] = logisticTest(x,w,y)
  numCorrect = 0;
  total = 0;
  classLabels = logisticClassify(x,w);
  for i = 1:length(y)
    if y(i) == classLabels(i)
      numCorrect = numCorrect+1;
    end
    total = total+1;
  end
  output = (numCorrect/total)*100;
end
