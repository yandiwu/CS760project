import pandas as pd
import numpy as np
from scipy.stats import norm

#Given a list of dicts of attributes and a new samples X,
#this method computes Prod Prob(x_j | y)
#For now, our attributes uses gaussian
#newX is an np array of size (n,k)
#This method returns an np array of size (n,)
def naiveProb(listOfAttributes,newX):
    probs = np.zeros(newX.shape)
    for n in range(len(listOfAttributes)):
        #print(listOfAttributes[n]['gaussmean'])
        #print(listOfAttributes[n]['gaussvar']**.5)
        #print(newX.dtype)
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
    nonInfoList = ['Group','GEOID','GeoName','TotalPop']
    dems = data.loc[data.Group == 'DEM'].copy().drop(columns=nonInfoList,inplace=False)
    reps = data.loc[data.Group == 'REP'].copy().drop(columns=nonInfoList,inplace=False)
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

    #Given the dem and rep "naive probabilities", we make predictions
    return ((demNaiveBayes-repNaiveBayes) >= 0).astype(int)

#data is a pandas dataframe, so this needs to be a little different
def naive_cross_val(data,folds=5, how = 'percent',order='ordered'):
    crossvaldata = data
    if (order == 'random'):
        crossvaldata = data.sample(frac=1)
    y = (crossvaldata['Group'] == 'DEM').to_numpy(int)
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
    natldata = pd.read_csv('CoveringClimateNow_2020_States_Districts_Only_Parties.csv')
    natldata.replace({'Group':{'Republican':'REP','Democratic':'DEM'}},inplace=True)
    newstates = natldata.loc[natldata.GeoType == 'State'].copy().drop(columns=['GeoType'])
    newcongs = natldata.loc[natldata.GeoType == 'cd116'].copy().drop(columns=['GeoType'])

    countydata = pd.read_csv('CoveringClimateNow_2020_County_Only.csv')
    newcounty = countydata.copy().drop(columns=['Margin'])
    newcounty.replace({'Group':{'republican':'REP','democratic':'DEM'}},inplace=True)

    stateframe = {'Region name':newstates['GeoName'],'Prediction':naive_cross_val(newstates,how='predictions')[0]}
    statepreds = pd.DataFrame(stateframe)
    congframe = {'Region name':newcongs['GeoName'],'Prediction':naive_cross_val(newcongs,how='predictions')[0]}
    congpreds = pd.DataFrame(congframe)
    countyframe = {'Region name':newcounty['GeoName'],'Prediction':naive_cross_val(newcounty,how='predictions')[0]}
    countypreds = pd.DataFrame(countyframe)

    statepreds.replace({'Prediction':{0.0:'REP',1.0:'DEM'}},inplace=True)
    congpreds.replace({'Prediction':{0.0:'REP',1.0:'DEM'}},inplace=True)
    countypreds.replace({'Prediction':{0.0:'REP',1.0:'DEM'}},inplace=True)

    naivebayespreds = statepreds.append([congpreds,countypreds],ignore_index = True)
    naivebayespreds.to_csv('naivebayespred.csv')
    
if __name__ == "__main__":
    main()
