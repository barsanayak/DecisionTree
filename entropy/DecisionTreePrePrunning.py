import pandas as pd
import numpy as np
import math

from sklearn.metrics import confusion_matrix
from matplotlib import pylab as plt
from sklearn.model_selection import train_test_split

'''
All the values related to the dataset are initialized here
'''
targetAttribute = 'Poisonous/Edible'
defaultTargetValue = 'p'
minimalGainThreshold = 0.145   # maxGainAttr:  spore-print-color    if  maxInfoGain > 0.144354902043, we get the data classified incorrectly

#targetAttribute = ''
targetValue = []
datasetOfInterest = 'MushroomDataSet_After_PreProcessing.xlsx'
test_size=0.2


# this method is used to calculate the entropy of the dataframs
def entropy(df):
    # targetDict holds unique values in form of dictionary
    targetDict = pd.value_counts(df[targetAttribute]).to_dict()
    dfLength = len(df)
    totalEntropy = 0
    # print(targetDict)
    for key, val in targetDict.items():
        multipleVal = val / dfLength
        totalEntropy += multipleVal * math.log2(multipleVal)
    return 0 - totalEntropy


#add values to targetValue
def initializeList(data):
    groupedData = data.groupby([targetAttribute]).groups.keys()
    return groupedData


def calcInforGainForEachAtrribute(df):
    # return maximum info gain
    maxInfoGain = 0;
    infoGainDict = {}
    attributes = list(df)
    totalEntropy = entropy(df)

    for att in attributes:
        # don't calculate gain for target attribute
        if (att == targetAttribute):
            continue
        grouped_data = df.groupby([att], as_index=False)
        totalRows = len(df)
        subEntropy = 0
        for key, value in grouped_data:
            eachGroupRows = len(value)
            S = entropy(value)
            # calculate |Sv/S|
            valEntrpy = eachGroupRows / totalRows
            # calculate |Sv/S|*Entropy(Sv)
            subEntropy = subEntropy + valEntrpy * S
        individualEntrpyGain = totalEntropy - subEntropy
        infoGainDict[att] = individualEntrpyGain
        # update the attribute with max information gain as per the maximu information gain
        if (individualEntrpyGain > maxInfoGain):
            maxInfoGain = individualEntrpyGain
            maxGainAttr = att;

    #Setting the minimalGainThreshold so that it stops comapring attributes with very less gain
    if (maxInfoGain < minimalGainThreshold):
        print("maxGainAttr: ", maxGainAttr)
        print("maxInfoGain :", maxInfoGain)
        # When the maximum information gain is very negligible dont retrun any attribute
        return None, None
    else:
        return maxGainAttr, maxInfoGain

#this method returns true if we reach a pure attribute.
def trueSet(s):
     return len(set(s)) == 1

#here we pass the attribute with highest gain each time and then calculate the subtree below based on rest attributes.
def createTree(df):

    x = df.drop(labels=targetAttribute, axis=1)
    y = df[targetAttribute]

    # If there could be no split, just return the original set
    if len(y) == 0:
        return y

    if trueSet(y):
        return set(y).pop()

    # We get the attribute (along with its gain) that gives the highest  information gain
    maxGainAttr, maxInfoGain = calcInforGainForEachAtrribute(df)
    # it means none of the values have a significant information gain => stop classifying to avoid overfitting
    if ((maxGainAttr == None) and (maxInfoGain == None)):
        return defaultTargetValue
    # Splitting the data using the  attribute with maximum gain
    sets = df.groupby(maxGainAttr)
    res = {}
    for k, v in sets:
        res["%s = %s" % (maxGainAttr, k)] = createTree(v)

    return res

# Now that we have got our decision tree in the form of dictionary we would predict the test data
def predictTarget(targetDict, test):
    for k, v in targetDict.items():
        keyList = k.split('=')
        col = keyList[0].strip()
        val = keyList[1].strip()
        if (test[col].iloc[0] == val):
            if k in targetDict:
                temp = targetDict.get(k)
                if (type(temp) == dict):
                    return predictTarget(v, test)
                else:
                    return v


# this method is used to print the confusion matrix
def printMatrix(y_test, predictedList, labels):
    cm = confusion_matrix(y_test, pd.Series(predictedList), labels)
    print(cm)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


# main method used to build the decision tre
def main():
    datasetAfterProcessing = pd.read_excel(datasetOfInterest)
    X = datasetAfterProcessing.drop(labels=targetAttribute, axis=1)
    y = datasetAfterProcessing[targetAttribute]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size, random_state=42)
    X_train[targetAttribute] = y_train
    targetValue = initializeList(X_train)
    targetDict = createTree(X_train)
    predictedList = []
    for i in list(range(0, len(X_test))):
        predictedVal = predictTarget(targetDict, X_test.iloc[[i]])
        predictedList.append(predictedVal)
    printMatrix(y_test, predictedList, list(targetValue))

# it enables the program execution from the maim method
if __name__ == "__main__": main()