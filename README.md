# CS760project: Climate Change and Partisan Politics
# Initial Data from 
Mildenberger, M., Marlon, J.R., Howe, P.D., & Leiserowitz, A. (2017) “The spatial distribution of Republican and Democratic climate opinions at state and local scales,” Climatic Change. https://doi.org/10.1007/s10584-017-2103-0.

County data: 
https://github.com/favstats/USElection2020-NYT-Results/tree/master/data/2020-11-07%2014-15-14?fbclid=IwAR0fo2FIUvczObmzUaNmnwNOnnuuCnnqfqxejy1dFlQP5sNmnd1ppcI0rKE

# Description of Files
1. KNN<br />
kNN_2020.py: Python code for the distance weighted k-Nearest Neighbors predictor, along with the local linear regression prediction for the margins of victory. <br />
2. Decision Trees <br />
DataParser.java: Java code for processing data to construct decision trees.<br />
DecTree.java: Java code for building decision trees and performing n-fold cross-validation. <br />
DecTreeNode.java: Java code to represent a single node in a tree. <br />
Project.java: Java code to run various methods in DecTree.java. <br />
3. Naive Bayes <br />
NaiveBayes2019.py: Python code using the naive bayes classifier to predict party affiliation for the 2019 data.<br />
NaiveBayes2020.py: Python code using the naive bayes classifier to make predictions for the 2020 US election.<br/ >


# Final Results Used for Analysis
States (Presidential): https://www.math.wisc.edu/~jenny/States/ <br />
Congressional Districts (House): https://www.math.wisc.edu/~jenny/CD <br />
Counties (Presidential): https://www.math.wisc.edu/~jenny/County 
