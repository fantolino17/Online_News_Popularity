%{
----------------------
Author: Frank Antolino
Date: May 2017
----------------------
_Description_:
Randomly selects points to use as training and testing data
Removes un-predictive features (columns 1 and 2)
Seperates data into features and class label
Returns matrix with trainData,trainClass,validData, validClass, testData,
testClass
%}

function output = sampleData(data)


  %Change class labels to binary, (numShares)
  for i = 1:length(data(:,end))
          if data(i,end)>=1400
            data(i,end) = 1;
          else
            data(i,end) = 0;
          end
  end


  %Sample data into training, validation, and testing
  numPoints = length(data(:,1));
  splitOne = round(numPoints*.6);
  splitTwo = round(numPoints*.8);
  randSeq = randperm(numPoints);
 
  %Remove features 1 and 2 (Non-Predictive)
  %Split into training, validation, and testing 60%,20%,20%, respectively 
  trainData = data(randSeq(1:splitOne) , 3:end-1);
  trainClass = data(randSeq(1:splitOne) , end);
  validData = data(randSeq(splitOne+1:splitTwo) , 3:end-1);
  validClass = data(randSeq(splitOne+1:splitTwo) , end);
  testData = data(randSeq(splitTwo+1:end) , 3:end-1);
  testClass = data(randSeq(splitTwo+1:end) , end);
  
  %Normalize data using zscores
  trainData = zscore(trainData);
  validData = zscore(validData);
  testData = zscore(testData);
  
  %Create object to return
  keySet = {'trainData','trainClass','validData','validClass','testData','testClass'};
  valueSet = {trainData,trainClass,validData,validClass,testData,testClass};
  output = containers.Map(keySet,valueSet);
  
end