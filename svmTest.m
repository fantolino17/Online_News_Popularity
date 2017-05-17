%{
----------------------
Author: Frank Antolino
Date: May 2017
----------------------
_Description_:
Parameters:
    predictedLabels: Model predicted class labels,
    actualLabels: Actual labels supplied by the dataset
Returns:
    Percent of actualLabels thats was correctly classified using 
    predictedLabels.

%}

function output = svmTest(predictedLabels, actualLabels)

  numCorrect = 0;
  total = 0;
  
  for i = 1:length(actualLabels)
    if actualLabels(i) == predictedLabels(i)
      numCorrect = numCorrect+1;
    end
    total = total+1;
  end
  output = numCorrect/total;
    
end