import pandas as pd
import numpy as np
from scipy.stats import norm

#Given a list of dicts of attributes and a new samples X,
#this method computes Prod Prob(x_j | y)
#For now, our attributes uses gaussian
#newX is an np array of size (n,6)
#This method returns an np array of size (n,)
def naiveProb(listOfAttributes,newX):
    probs = np.zeros(newX.shape)
    for n in range(len(listOfAttributes)):
        probs[:,n] = norm.pdf(newX[:,n],loc = listOfAttributes[n]['gaussmean'],scale = listOfAttributes[n]['gaussvar']**.5)
    return probs.prod(axis = 1)

#data is a pandas series
#Computes the sample mean and sample (unbiased) variance
#Returns a dict of the form {'gaussmean': mean, 'gaussvar':var}
def gaussian(data):
    info = {}
    info['gaussmean'] = data.mean()
    info['gaussvar'] = data.var()
    return info
#data is a pandas dataframe containing the data and the labels
#newX is a pandas dataframe of size (n,k) containing n samples to be predicted
#This method returns an np array of predictions of size (n,)
def naiveBayesPredictor(data,newX):
    #List of features that aren't survey data
    nonInfoList = ['GeoType','Group','GEOID','GeoName','TotalPop']
    dems = data.loc[data.Group == 'Dem'].copy().drop(columns=nonInfoList,inplace=False)
    reps = data.loc[data.Group == 'Rep'].copy().drop(columns=nonInfoList,inplace=False)
    toPredict = newX.copy().drop(columns=nonInfoList,inplace=False)
    demProb = len(dems) / len(data)
    repProb = len(reps)/len(data)
    demList = []
    repList = []
    
    for col in dems:
        demList.append(gaussian(dems[col]))
        repList.append(gaussian(reps[col]))
    #Prob(y) * Product of Prob(x | y) that we use in Naive Bayes
    demNaiveBayes = demProb * naiveProb(demList,toPredict.to_numpy())
    repNaiveBayes = repProb * naiveProb(repList,toPredict.to_numpy())

    #Given the dead and survival "naive probabilities", we make predictions
    return ((demNaiveBayes-repNaiveBayes) >= 0).astype(int)

#data is a pandas dataframe, so this needs to be a little different
def naive_cross_val(data,folds=5, how = 'percent'):
    #randomly permute the rows
    crossvaldata = data.sample(frac=1,random_state = 42)
    y = (crossvaldata['Group'] == 'Dem').to_numpy(int)
    predictions = np.zeros(crossvaldata.shape[0])
    for n in range(folds):
        if (n == 0):
            newData = crossvaldata.iloc[data.shape[0]//folds:,:]
            dataToPredict = crossvaldata.iloc[:data.shape[0]//folds,:]
            predictions[:data.shape[0]//folds] = naiveBayesPredictor(newData,dataToPredict)
        elif (n == (folds - 1)):
            newData = crossvaldata.iloc[:((folds-1)*data.shape[0])//folds,:]
            dataToPredict = crossvaldata.iloc[((folds-1)*data.shape[0])//folds:,:]
            predictions[((folds-1)*data.shape[0])//folds:] = naiveBayesPredictor(newData,dataToPredict)
        else:
            end = (n *data.shape[0]) // folds
            start = ((n+1) * data.shape[0]) // folds
            newData = crossvaldata.iloc[:end,:].append(crossvaldata.iloc[start:,:],ignore_index=True)
            dataToPredict = crossvaldata.iloc[end:start,:]
            predictions[end:start] = naiveBayesPredictor(newData,dataToPredict)
    if (how == 'percent'):
        numwrong = np.sum(np.abs(y - predictions))
        percentwrong = numwrong / crossvaldata.shape[0]
        return (1-percentwrong)
    if (how == 'predictions'):
        return (predictions,y,crossvaldata)

def main():
    data = pd.read_excel('/Users/Solly/Desktop/CS760/760Project/projectdata.xlsx')
    print(naive_cross_val(data,folds=5,how='percent'))

if __name__ == "__main__":
    main()