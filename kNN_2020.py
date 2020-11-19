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

#Wisconsin
x = [74.203,25.5,76.332,22.162,76.451,22.249,76.501,22.227,76.465,21.852,74.521,23.666,77.678,21.309,73.087,25.435,76.163,23.083,74.63,24.332]
y = [72.312, 27.188, 74.098, 24.617, 75.784, 23.212,75.494,	23.094,	75.522,	23.26,	72.948,	25.044,	76.1, 22.214, 72.179, 25.549, 73.924, 24.376, 72.597, 26.256]
N1 = nearest_neighbors(x, Xstate, Ystate, 8)
N2 = nearest_neighbors(y, Xstate, Ystate, 8)
#print(locally_weighted_regression(x, N1[1], N1[2]))
#print(locally_weighted_regression(x, N2[1], N2[2]))
#z = [72.614, 26.464, 75.775, 22.087, 77.569, 21.098, 76.304, 22.015, 75.923, 22.259, 76.013, 22.6, 77.631, 20.586, 73.266, 25.529, 74.968, 23.63, 75.932, 22.731]
#N3 = nearest_neighbors(z, Xdistrict, Ydistrict, 8)
#print(N3)
#print(locally_weighted_regression(x, N3[1], N3[2]))

##############################################################################
########## FIVE FOLD CROSS VALIDATION #########################################
##############################################################################

def five_fold_cross_validation(X, Y):
    """
    X, the np array with all the training data
    Y, the labels of the training data
    """
    accuracy = 0
    for i in range(5):
        Xtrain = deepcopy(X)
        ytrain = deepcopy(Y)
        Xtest = deepcopy(X)
        ytest = deepcopy(Y)
        r = math.floor(len(X)/5)
        d = len(X)%5 #last set will be a bit bigger
        if i != 4:
            Xtest = Xtest[i*r : (i + 1)*r] #training data
            ytest = ytest[i*r : (i + 1)*r] #testing data
        #account for the fact that there are 51 states
        else:
            Xtest = Xtest[i*r : 5*r+d]
            ytest = ytest[i*r : 5*r+d]
        #delete the testing data from the training data
        j = 0
        while j < r:
            Xtrain = np.delete(Xtrain, i*r, 0)
            ytrain = np.delete(ytrain, i*r, 0)
            j += 1
        if i == 4:
            if d != 0:
                Xtrain = np.delete(Xtrain, 4*r, 0)
                ytrain = np.delete(ytrain, 4*r, 0)
        accuracy_count = 0
        for l in range(len(Xtest)): #counting the number of accurately predicted cases
            yhatlist = []
            yhat = kNN_accuracy(Xtest[l], Xtrain, ytrain)
            if yhat == ytest[l]:
                accuracy_count += 1
        accuracy += accuracy_count/(len(Xtest))
    return accuracy/5

def kNN_accuracy(x, Xtrain, Ytrain):
    """
    Helper function for calculating the prediction
    x, a feature vector to predict the election results of
    Xtrain, an np array of the training data
    Ytrain, an np array of the training labels
    """
    k = round(math.sqrt(len(Xtrain)*(1.25))) #calculate the number of nearest neighbors
    data = nearest_neighbors(x, Xtrain, Ytrain, k)
    return locally_weighted_regression(x, data[1], data[2])

#perform 5-fold cross validation on state data
print(five_fold_cross_validation(Xstate, Ystate))

#randomly shuffle the congressional data to make it more balanced
z = list(zip(Xdistrict, Ydistrict))
random.shuffle(z)
Xdistrict, Ydistrict = zip(*z)

#perform 5-fold cross validation on district data
print(five_fold_cross_validation(Xdistrict, Ydistrict))
