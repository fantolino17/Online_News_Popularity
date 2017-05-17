%{
----------------------
Author: Frank Antolino
Date: May 2017
----------------------
_Description_: See README.txt

%}

 all = importdata('OnlineNewsPopularity.csv');
 feats = all.data;
  
 %SampleData removes un-predictive features
 %Turns class label binary (<1400 -> 0 , >1400 -> 1)
 %Splits data into training, validation, testing data (60,20,20)
 %Normalizes data using zscores
 dataObj = sampleData(feats);
 
 trainData = dataObj('trainData');
 trainClass = dataObj('trainClass');
 validData = dataObj('validData');
 validClass = dataObj('validClass');
 testData = dataObj('testData');
 testClass = dataObj('testClass');

 
 %featSelectFisher selects attributes with highest Fisher scores until 
 %classifier performance decreases.
 topFeats = featSelectFisher(trainData,trainClass,testData,testClass);
 disp('The features selected via Fisher Criterion are: ' )
 disp(topFeats)
 
 
 %Weight Vector
 sz = length(trainData(1,topFeats));
 w0 = zeros(1,sz+1);%Plus one for offset
 
 %Select topFeats only, and add one for offset
 trainData = [ trainData(:,topFeats) , ones(length(trainData), 1)];
 testData = [ testData(:,topFeats) , ones(length(testData), 1)];
 validData = [ validData(:,topFeats) , ones(length(validData), 1)];

 %-----------------------------------------------------------------
 %Logistic Classification
 %-----------------------------------------------------------------
 
 %Test learning with different parameters:
 %-Number of loops
 %-epislon(stepsize)
 %-lambda (for L2)
 disp('--------------------------------------------')
 disp('Using Logistic Classification')
 disp('Varying hyper-parameters as described below')
 disp('-------------------------------------------')
 
 
 %-------------NumLoops=10, Vary epsilon, no L2 (lambda=max)---------
 wNew = learnLogisticWeights(w0,trainData,trainClass, 10, .01, realmax);
 disp('NumLoops = 10    EPS = .01    Lambda = inf')
 acc = logisticTest(validData,wNew,validClass);
 disp(['Accuracy:' , num2str(acc)]);
 disp(' ')

 wNew = learnLogisticWeights(w0,trainData,trainClass, 10, .02, realmax);
 disp('NumLoops = 10    EPS = .02    Lambda = inf')
 acc = logisticTest(validData,wNew,validClass);
 disp(['Accuracy:' , num2str(acc)]);
 disp(' ')
 
 wNew = learnLogisticWeights(w0,trainData,trainClass, 10, .03, realmax);
 disp('NumLoops = 10    EPS = .03    Lambda = inf')
 acc = logisticTest(validData,wNew,validClass);
 disp(['Accuracy:' , num2str(acc)]);
 disp(' ')

 wNew = learnLogisticWeights(w0,trainData,trainClass, 10, .04, realmax);
 disp('NumLoops = 10    EPS = .04    Lambda = inf')
 acc = logisticTest(validData,wNew,validClass);
 disp(['Accuracy:' , num2str(acc)]);
 disp(' ')

 wNew = learnLogisticWeights(w0,trainData,trainClass, 10, .05, realmax);
 disp('NumLoops = 10    EPS = .05    Lambda = inf')
 acc = logisticTest(validData,wNew,validClass);
 disp(['Accuracy:' , num2str(acc)]);
 disp(' ')
 
 %----------NumLoops=10, epsilon=.01, Vary lambda in L2 reg----------

 wNew = learnLogisticWeights(w0,trainData,trainClass, 10, .01, 100);
 disp('NumLoops = 10    EPS = .01    Lambda = 100')
 acc = logisticTest(validData,wNew,validClass);
 disp(['Accuracy:' , num2str(acc)]);
 disp(' ')
 
 wNew = learnLogisticWeights(w0,trainData,trainClass, 10, .01, 50);
 disp('NumLoops = 10    EPS = .01    Lambda = 50')
 acc = logisticTest(validData,wNew,validClass);
 disp(['Accuracy:' , num2str(acc)]);
 disp(' ')
 
 wNew = learnLogisticWeights(w0,trainData,trainClass, 10, .01, 30);
 disp('NumLoops = 10    EPS = .01    Lambda = 30')
 acc = logisticTest(validData,wNew,validClass);
 disp(['Accuracy:' , num2str(acc)]);
 disp(' ')
 
 wNew = learnLogisticWeights(w0,trainData,trainClass, 10, .01, 10);
 disp('NumLoops = 10    EPS = .01    Lambda = 10')
 acc = logisticTest(validData,wNew,validClass);
 disp(['Accuracy:' , num2str(acc)]);
 disp(' ')
 
 wNew = learnLogisticWeights(w0,trainData,trainClass, 10, .01, 5);
 disp('NumLoops = 10    EPS = .01    Lambda = 5')
 acc = logisticTest(validData,wNew,validClass);
 disp(['Accuracy:' , num2str(acc)]);
 disp(' ')
 
 %-------------Vary NumLoops, epsilon=.01, no L2 (lambda=max)---------

 wNew = learnLogisticWeights(w0,trainData,trainClass, 5, .01, realmax);
 disp('NumLoops = 5    EPS = .01    Lambda = inf')
 acc = logisticTest(validData,wNew,validClass);
 disp(['Accuracy:' , num2str(acc)]);
 disp(' ')
 
 wNew = learnLogisticWeights(w0,trainData,trainClass, 10, .01, realmax);
 disp('NumLoops = 10    EPS = .01    Lambda = inf')
 acc = logisticTest(validData,wNew,validClass);
 disp(['Accuracy:' , num2str(acc)]);
 disp(' ')
 
 wNew = learnLogisticWeights(w0,trainData,trainClass, 25, .01, realmax);
 disp('NumLoops = 25    EPS = .01    Lambda = inf')
 acc = logisticTest(validData,wNew,validClass);
 disp(['Accuracy:' , num2str(acc)]);
 disp(' ')
 
 wNew = learnLogisticWeights(w0,trainData,trainClass, 50, .01, realmax);
 disp('NumLoops = 50    EPS = .01    Lambda = inf')
 acc = logisticTest(validData,wNew,validClass);
 disp(['Accuracy:' , num2str(acc)]);
 disp(' ')
 
 wNew = learnLogisticWeights(w0,trainData,trainClass, 100, .01, realmax);
 disp('NumLoops = 100    EPS = .01    Lambda = inf')
 acc = logisticTest(validData,wNew,validClass);
 disp(['Accuracy:' , num2str(acc)]);
 disp(' ')
 
 %----------------------------------------------------------------------
 %End Logistic Classification
 %----------------------------------------------------------------------
 
 
 %----------------------------------------------------------------------
 %Begin Support Vector Machines
 %----------------------------------------------------------------------

 %Test different hyper-parameters:
 %-Kernel Function
 %-c for slack variables
 %-Iteration Limit
 disp('-------------------------------------------')
 disp('Using SVM Classification')
 disp('Varying hyper-parameters as described below')
 disp('-------------------------------------------')
 disp('')
 
 
 %------------------------Alter c for slack variables----------------------
 disp('Kernel Function = RBF    c = .0001    Iteration Limit = None')
 Mdl = fitcsvm(trainData,trainClass, 'KernelFunction', 'rbf', 'BoxConstraint' , .01 );
 predictedLabels = predict(Mdl,testData);
 acc = svmTest(predictedLabels, testClass);
 disp(['Accuracy:' , num2str(acc)]);
 disp(' ')

 disp('Kernel Function = RBF    c = .1    Iteration Limit = None')
 Mdl = fitcsvm(trainData,trainClass, 'KernelFunction', 'rbf', 'BoxConstraint' , .1 );
 predictedLabels = predict(Mdl,testData);
 acc = svmTest(predictedLabels, testClass);
 disp(['Accuracy:' , num2str(acc)]);
 disp(' ')

 disp('Kernel Function = RBF    c = 1    Iteration Limit = None')
 Mdl = fitcsvm(trainData,trainClass, 'KernelFunction', 'rbf', 'BoxConstraint' , 1 );
 predictedLabels = predict(Mdl,testData);
 acc = svmTest(predictedLabels, testClass);
 disp(['Accuracy:' , num2str(acc)]);
 disp(' ')
 
 disp('Kernel Function = RBF    c = 10    Iteration Limit = None')
 Mdl = fitcsvm(trainData,trainClass, 'KernelFunction', 'rbf', 'BoxConstraint' , 10 );
 predictedLabels = predict(Mdl,testData);
 acc = svmTest(predictedLabels, testClass);
 disp(['Accuracy:' , num2str(acc)]);
 disp(' ')
 
 disp('Kernel Function = RBF    c = 100    Iteration Limit = None')
 Mdl = fitcsvm(trainData,trainClass, 'KernelFunction', 'rbf', 'BoxConstraint' , 100 );
 predictedLabels = predict(Mdl,testData);
 acc = svmTest(predictedLabels, testClass);
 disp(['Accuracy:' , num2str(acc)]);
 disp(' ')
 
 %--------------------------Alter Kernel Function--------------------------
 
 disp('Kernel Function = Linear    c = .001    Iteration Limit = 1e4')
 Mdl = fitcsvm(trainData,trainClass, 'KernelFunction', 'linear', 'BoxConstraint' , .001, 'IterationLimit', 1e4 );
 predictedLabels = predict(Mdl,testData);
 acc = svmTest(predictedLabels, testClass);
 disp(['Accuracy:' , num2str(acc)]);
 disp(' ')
 
 
 disp('Kernel Function = RBF    c = .001    Iteration Limit = 1e4')
 Mdl = fitcsvm(trainData,trainClass, 'KernelFunction', 'rbf', 'BoxConstraint' , .001, 'IterationLimit', 1e4 );
 predictedLabels = predict(Mdl,testData);
 acc = svmTest(predictedLabels, testClass);
 disp(['Accuracy:' , num2str(acc)]);
 disp(' ')
 
 
 disp('Kernel Function = Polynomial, Order 2    c = .001    Iteration Limit = 1e4')
 Mdl = fitcsvm(trainData,trainClass, 'KernelFunction', 'polynomial', 'PolynomialOrder', 2,  'BoxConstraint' , .001, 'IterationLimit', 1e4 );
 predictedLabels = predict(Mdl,testData);
 acc = svmTest(predictedLabels, testClass);
 disp(['Accuracy:' , num2str(acc)]);
 disp(' ')

 disp('Kernel Function = Polynomial, Order 3    c = .001    Iteration Limit = 1e4')
 Mdl = fitcsvm(trainData,trainClass, 'KernelFunction',  'polynomial', 'PolynomialOrder', 3, 'BoxConstraint' , .001, 'IterationLimit', 1e4 );
 predictedLabels = predict(Mdl,testData);
 acc = svmTest(predictedLabels, testClass);
 disp(['Accuracy:' , num2str(acc)]);
 disp(' ')
 
 disp('Kernel Function = Polynomial, Order 4    c = .001    Iteration Limit = 1e4') 
 Mdl = fitcsvm(trainData,trainClass, 'KernelFunction',  'polynomial', 'PolynomialOrder', 4, 'BoxConstraint' , .001, 'IterationLimit', 1e4 );
 predictedLabels = predict(Mdl,testData);
 acc = svmTest(predictedLabels, testClass);
 disp(['Accuracy:' , num2str(acc)]);
 disp(' ')
 
 
 %--------------------------------Alter Iteration--------------------------
 
 disp('Kernel Function = RBF    c = .001    Iteration Limit = 1e4')
 Mdl = fitcsvm(trainData,trainClass, 'KernelFunction', 'rbf', 'BoxConstraint' , .001, 'IterationLimit', 1e4 );
 predictedLabels = predict(Mdl,testData);
 acc = svmTest(predictedLabels, testClass);
 disp(['Accuracy:' , num2str(acc)]);
 disp(' ')
 
 disp('Kernel Function = RBF    c = .001    Iteration Limit = 1e5')
 Mdl = fitcsvm(trainData,trainClass, 'KernelFunction', 'rbf', 'BoxConstraint' , .001, 'IterationLimit', 1e5 );
 predictedLabels = predict(Mdl,testData);
 acc = svmTest(predictedLabels, testClass);
 disp(['Accuracy:' , num2str(acc)]);
 disp(' ')
 
 disp('Kernel Function = RBF    c = .001    Iteration Limit = 1e6')
 Mdl = fitcsvm(trainData,trainClass, 'KernelFunction', 'rbf', 'BoxConstraint' , .001, 'IterationLimit', 1e6 );
 predictedLabels = predict(Mdl,testData);
 acc = svmTest(predictedLabels, testClass);
 disp(['Accuracy:' , num2str(acc)]);
 disp(' ')
 
 disp('Kernel Function = RBF    c = .001    Iteration Limit = 1e7')
 Mdl = fitcsvm(trainData,trainClass, 'KernelFunction', 'rbf', 'BoxConstraint' , .001, 'IterationLimit', 1e7 );
 predictedLabels = predict(Mdl,testData);
 acc = svmTest(predictedLabels, testClass);
 disp(['Accuracy:' , num2str(acc)]);
 disp(' ')
 
 disp('Kernel Function = RBF    c = .001    Iteration Limit = 1e8')
 Mdl = fitcsvm(trainData,trainClass, 'KernelFunction', 'rbf', 'BoxConstraint' , .001, 'IterationLimit', 1e8 );
 predictedLabels = predict(Mdl,testData);
 acc = svmTest(predictedLabels, testClass);
 disp(['Accuracy:' , num2str(acc)]);
 disp(' ')
 
 
 %------------------------------------------------------------------------
 %End Support Vector Machines
 %------------------------------------------------------------------------
 
 %------------------------------------------------------------------------
 %End of File
 %------------------------------------------------------------------------
 
 