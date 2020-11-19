import csv
import numpy as np
from copy import deepcopy

#import csv file as matrix
reader = csv.reader(open("partisan_map_data.csv", "rt"), delimiter=",")
next(reader, None)
r = list(reader)
result = np.array(r).astype("float")

#create X and Y, matrices of training data and labels
Y = np.delete(result, np.s_[1::], 1)
Y1 = deepcopy(Y)
Y2 = deepcopy(Y)
result = np.delete(result, 0, 1) #delete first column- not a feature (these are the labels)
result = np.delete(result, 0, 1) #delete second column- not relevant feature (regional code)
result = np.delete(result, 0, 1) #delete third column- not relevant feature (population)
D = len(result[0])
X1 = deepcopy(result)
Xstate = result[:102] #first rows are state data
Xdistricts = result[102:] #congressional district data
#beta = np.ones((D, 1))
#Xstate = np.hstack((beta, Xstate)) #augment the training data by adding intercepts
#Xdistricts = np.hstack((beta, Xdistricts)) #augment the training data by adding intercepts

#create matrices of labels of training data
Ystate = Y1[:102]
Ydistricts = Y2[102:]
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
        #closest_distances += [manhattan_distance(X[l], x)]
        closest_distances += [np.linalg.norm(X[l] - x)]
        Xk[l] = X[l]
        closest_labels += [Y[l]]
    for i in range(D):
        #d = manhattan_distance(X[i], x)
        d = np.linalg.norm(X[i] - x)
        for m in range(k):
            if closest_distances[m] > d: #if some distance in the list is greater than current distance, replace it
                closest_distances[m] = d
                Xk[m] = X[i]
                closest_labels[m] = Y[i] #add the ith label to the list of labels
                break #no need to replace any of the other closest distances
    return (Xk, closest_labels)

def manhattan_distance(a, b):
    """
    Calculates the manhattan (L1) distance between two vectors a and b
    """
    sum = 0
    l = len(a)
    for i in range(l):
        sum += abs(a[i] - b[i])
    return sum

#Virginia
x = [84.518,5.73,86.017,5.037,82.56,5.47,90.142,8.415,82.484,16.321,93.211,5.668,59.362,39.88,84.308,13.709,93.369,6.026,91.131,2.375,78.648,14.94,78.465,12.573,84.123,15.759,79.235,11.542,85.268,2.128]
#y = [1, 53, 5661460, 72.758, 26.077, 78.133, 19.364, 79.458, 19.232, 78.023, 19.936, 79.564, 18.463, 78.68, 19.748, 80.391, 17.677, 78.256, 20.635, 78.167, 20.168, 79.062, 20.049]
y = [79.61,	8.735, 81.93, 7.786, 79.541, 7.159,	88.379,	9.942, 74.719, 24.953, 89.5, 7.126,	52.907,	46.092,	79.853,	17.671,	90.677,	8.682,	87.898,	3.635, 72.399, 20.481, 64.427, 17.052,	80.152,	19.821,	77.076,	13.569,	79.863,	2.746]
z = [48.749, 19.549, 60.825, 12.67, 53.861,	15.361,	72.479,	25.164,	57.763,	40.27,	84.527,	14.313,	27.469,	73.38,	54.456,	44.277,	64.083,	35.81,	61.034,	22.199,	42.831,	41.883,	41.906,	42.129,	48.549,	51.495,	44.951,	46.972,	51.75,	9.07]
#print(nearest_neighbors(x, Xstate, Ystate, 7))
#print(nearest_neighbors(y, Xdistricts, Ydistricts, 7))
print(nearest_neighbors(z, Xdistricts, Ydistricts, 7))
#print(nearest_neighbors(y, Xstate, Ystate, 7))
#print(len(y))
