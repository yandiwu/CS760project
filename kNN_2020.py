import csv
import numpy as np
import math
import random
from copy import deepcopy

#import csv file as matrix
reader = csv.reader(open("CoveringClimateNow_2020_webdata.csv", "rt"), delimiter=",")
next(reader, None) #skip the first row
next(reader, None) #skip the second row (national data)
r = list(reader)
result = np.array(r).astype("float")

#create X and Y, matrices of training data and labels
Y = np.delete(result, np.s_[1::], 1)
Y1 = deepcopy(Y)
Y2 = deepcopy(Y)
result = np.delete(result, 0, 1) #delete first column- not a feature (these are the labels)
result = np.delete(result, 0, 1) #delete second column- not relevant feature (regional code)
D = len(result[0])
X1 = deepcopy(result)
Xstate = result[:51] #first rows are state data
Xdistrict = X1[3193:] #congressional district data

#create matrices of labels of training data
Ystate = Y1[:51]
Ydistrict = Y2[3193:]
D = len(result)

#import county data csv file as matrix
reader2 = csv.reader(open("CoveringClimateNow_2020_County_Only.csv", "rt"), delimiter=",")
next(reader2, None) #skip the first row
r2 = list(reader2)
Xcounty = np.array(r2).astype("float")

#create X and Y, matrices of training data and labels on the county level
Ycounty = np.delete(Xcounty, np.s_[1::], 1) #first column contains all the county labels
Ycounty1 = deepcopy(Y)
Xcounty = np.delete(Xcounty, 0, 1) #delete first column- not a feature (these are the labels)
Ymargins = np.delete(Xcounty, np.s_[1::], 1) #second column contains all the margins
Xcounty = np.delete(Xcounty, 0, 1) #delete second column- not feature (margins)
D = len(Xcounty[0])

#k nearest neighbors
def nearest_neighbors(x, X, Y, k):
    """
    Gives the k nearest neighbors and their labels, given a feature vector and training data/labels
    Args:
    x, the feature vector we are trying to predict
    X, the training data feature matrix
    Y, the training data labels
    k: the number of neighbors used in the algorithm

    Returns:
    Xk: a k by d numpy array, the feature matrix (d = number of features) with data on the k nearest neighbors
    closest_labels: a k by 1 numpy array containing all the labels associated with the k nearest neighbors
    """
    n = len(X[0])
    D = len(X)
    closest_distances = []
    closest_labels = []
    Xk = np.empty(shape=(k, n), dtype=float)
    #initiate list of closest distances by computing the first k distances
    for l in range(k): #add the first k entries to all the lists initiated to prevent index of out bounds errors
        closest_distances += [np.linalg.norm(X[l] - x)]
        Xk[l] = X[l]
        closest_labels += [Y[l]]
    for i in range(D):
        d = np.linalg.norm(X[i] - x)
        for m in range(k):
            if closest_distances[m] > d: #if some distance in the list is greater than current distance, replace it
                closest_distances[m] = d
                Xk[m] = X[i]
                closest_labels[m] = Y[i] #add the ith label to the list of labels
                break #no need to replace any of the other closest distances
    return (Xk, closest_distances, closest_labels)

def locally_weighted_regression(x, closest_distances, neighbor_labels):
    """
    Locally weighted regression algorithm
    Args:
    x, the feature vector we are trying to classify
    neighbors, the nearest nearest neighbors
    neighbor_labels, the labels of the nearest neighbors

    Returns:
    yhat, 0 for Rep and 1 for Dem
    """
    denom = 0
    num = 0
    k = len(closest_distances) #number of nearest neighbors
    for i in range(1, k):
        w = 1/(closest_distances[i])
        denom += w
        num += w * neighbor_labels[i]
    if num/denom > 0.5:
        return 1
    else:
        return 0

##############################################################################
########## N-FOLD CROSS VALIDATION #########################################
##############################################################################

def cross_validation(X, Y, n):
    """
    X, the np array with all the training data
    Y, the labels of the training data
    n, either 5 or 10 depending on the number of groups we need the split the data into
    """
    accuracy = 0
    k = round(math.sqrt(len(X)))
    for i in range(n):
        Xtrain = deepcopy(X)
        ytrain = deepcopy(Y)
        Xtest = deepcopy(X)
        ytest = deepcopy(Y)
        r = math.floor(len(X)/n)
        d = len(X)%n #last set will be a bit bigger
        if i != n - 1:
            Xtest = Xtest[i*r : (i + 1)*r] #training data
            ytest = ytest[i*r : (i + 1)*r] #testing data
        #account for the fact that there are 51 states
        else:
            Xtest = Xtest[i*r : n*r+d]
            ytest = ytest[i*r : n*r+d]
        #delete the testing data from the training data
        j = 0
        while j < r:
            Xtrain = np.delete(Xtrain, i*r, 0)
            ytrain = np.delete(ytrain, i*r, 0)
            j += 1
        if i == n - 1:
            if d != 0:
                Xtrain = np.delete(Xtrain, (n - 1)*r, 0)
                ytrain = np.delete(ytrain, (n - 1)*r, 0)
        accuracy_count = 0
        for l in range(len(Xtest)): #counting the number of accurately predicted cases
            yhatlist = []
            yhat = kNN_accuracy(Xtest[l], Xtrain, ytrain, k)
            if yhat == ytest[l]:
                accuracy_count += 1
        accuracy += accuracy_count/(len(Xtest))
    return accuracy/n

def kNN_accuracy(x, Xtrain, Ytrain, k):
    """
    Helper function for calculating the prediction the prediction for N-fold cross validation. 
    x, a feature vector to predict the election results of
    Xtrain, an np array of the training data
    Ytrain, an np array of the training labels
    k, the number of nearest neighbors to find
    """
    data = nearest_neighbors(x, Xtrain, Ytrain, k)
    return locally_weighted_regression(x, data[1], data[2])

#perform 5-fold cross validation on state data
print(cross_validation(Xstate, Ystate, 5))

#average accuracies of 10 runs of 5-fold cross validation at the electoral district level
accuracy = 0
for _ in range(10):
    #randomly shuffle the congressional data to make it more balanced
    z = list(zip(Xdistrict, Ydistrict))
    random.shuffle(z)
    Xdistrict, Ydistrict = zip(*z)
    #perform 5-fold cross validation on district data
    accuracy += cross_validation(Xdistrict, Ydistrict, 5)
print(accuracy/10)

#perform 10-fold cross validation on county data
z1 = list(zip(Xcounty, Ycounty))
random.shuffle(z1)
Xcounty, Ycounty = zip(*z1)
print(cross_validation(Xcounty, Ycounty, 10))
