Author: Frank Antolino
Date: May 2017

    Running the file run.m requires having the 'OnlineNewsPopularity.csv' 
file provided by UCI. This data is imported, and run.m calls the sampledata
function to format and split up the data into training, testing, and
validation sets. It then calls the featFisherSelect function which will
call fisherScores. FisherScores returns all scores for all features, and
featFisherSelect will add the highest scored features until performance
drops, and return this subset of all features.
    Then, run.m calls several functions (written by Frank Antolino),
pertaining to logisitic classification, 15 times, altering the parameters
passed each time. The results are continually logged to the console.
    Similarly, 15 sets of parameters are passed several SVM functions,
(from the Statistics and Machine Learning Toolbox provided by MATLAB), and
their results are continually logged to the console as well."# Online_News_Popularity" 
