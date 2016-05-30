## Feature generation
Work Distribution: SVM-May, Random Forest-Erica, Adaboost-Yilin

FeatureGenerator.py: Python file that generates the data set we used in our analysis and prediction.

Fraud_score.py: function that calculates the fraud score of device, country, auction and merchandise.

##Input File

features.csv: file for input

##Classification Algorithms and Outputs:

randomforests.py: Random Forests Algorithm for classification
#Install:
pip install sklearn
pip install numpy
pip install csv
pip install copy


submission_rf.cv: prediction result of Random forests algorithm

main.R: main function that generate the final submission csv.file using Adaboost. 

Decisiontree.R: function that implements classification tree and generate 1000 weak learners. Called by main.R

Adaboost.R: function of Adaboost algorithm. Called by main.R

Adaboost.csv: prediction result of Adaboost algorithm.

SVM.m: main function that generate the final submission csv.file using SVM.

svm_rbf_poly.m: function implements svm with rbf and polynomial kernel. Called by SVM.m

rbf_kernel.m: function that applies the RBF kernel on the input. Called by svm_rbf_poly.m

svm.csv: prediction result of SVM algorithm.

Resulting file with best score.csv: Submission file to Kaggle with best score

Report.pdf: Brief report of the project
