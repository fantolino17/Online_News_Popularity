%{
----------------------
Author: Frank Antolino
Date: May 2017
----------------------
_Description_:
Returns a matrix of fisher scores and  corresponding indeces
sortedScores(1,:) is all scores.
sortedScores(2,:) is the feature indeces for these scores.
The matrix will be sorted in descending order, according to
Fisher Scores. (Higher Scores imply more discriminative feature)
%}

function sortedScores = fisherScores(feats, class)

    sz = length(feats(1,:));%sz is number of feats
    scores = zeros(2,sz);
    
    sumZero = 0;%Number of instances of class 0
    sumOne = 0;%Number of instances of class 1
    
    for j = 1:length(class)
       if class(j) == 0
           sumZero = sumZero+1;
       elseif class(j) == 1
           sumOne = sumOne+1;
       end
    end
            
    for i = 1:length(feats(1,:))%each feature
       
       muTotal = mean(feats(:,i));%Mean value of data for a feature (in both classes)
       vctZero = feats(find(class==0), i);%Get feature values where class equals 0
       vctOne = feats(find(class==1), i);%Get feature values where class equals 1
       muZero = mean(vctZero);%Mean value for feature i, class 0 
       stdZero = std(vctZero);%STD value for feature i, class 0
       muOne = mean(vctOne);%Mean value for feature i, class 1
       stdOne = std(vctOne);%STD value for feature i, class 1
       
       classZeroNum = sumZero*((muZero - muTotal)^2);
       classZeroDen = sumZero*(stdZero^2);
       classOneNum = sumOne*((muOne - muTotal)^2);
       classOneDen = sumOne*(stdOne^2);
       
       scores(1,i) = (classZeroNum + classOneNum) / (classZeroDen+classOneDen);
       scores(2,i) = i;
    end
    
     [Y,I] = sort(scores(1,:) , 'descend');
     sortedScores = scores(:,I);
end
