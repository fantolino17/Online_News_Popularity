%{
----------------------
Author: Frank Antolino
Date: May 2017
----------------------
_Description_:
This functions calls fisherScores, which returns a sorted matrix of
fisher scores and indeces. This function then linearly loops through 
all the sorted features, and picks the subset that has the highest 
classification accuracy on the test data.
_Disclaimer_: 
This function assumes fisher scores in descending order will
continually be less and less discriminatory, and so instead of checking ALL
possible subsets, in order to greatly reduce runtime, the function simply 
adds features one at a time, with each one being 'less and less informative'
and picks the subset with the greatest accuracy. This runs in O(n) where n
equals the number of features. (As opposed to O(n!) if we checked all 
possible subsets) Furthermore, it uses a logistic classifier with 10
iterations, step-size (epsilon) of .001, and no regularization to compare
accuracies.
%}

function output = featSelectFisher(trainData, trainClass, testData, testClass)
  
  %Get FisherScores, sorted from best feats to worst.
  sortedScores = fisherScores(trainData, trainClass)
  
  %create weight vector, initialized to zeros
  sz = length(trainData(1,:));
  w0 = zeros(1,sz+1);
  trainData = [trainData ones(length(trainData), 1)]; %add one for offset
  wNew = learnLogisticWeights(w0,trainData, trainClass, 10, .001, realmax);
  newResult = zeros(2,sz); 
  
  %Continually add feature and test classifier.
  %Each next feature should be less informative (Smaller score)
  for i=1:length(sortedScores)
      
    new = logisticTest(testData(: , sortedScores(2 , 1:i)), wNew(sortedScores(2, 1:i)),testClass);
    newResult(1,i) =  new; 
    newResult(2,i) = i;

  end
  %Pick which subset of features performed the best
  %Returns top X features to use
  maxVal = max(newResult(1,:));
  ind = find(maxVal==newResult(1,:));
  top = newResult(2, ind);
  output = sortedScores(2,1:top);
end