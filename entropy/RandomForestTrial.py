# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 14:32:24 2017

@author: nithi
"""

import pandas as pd
import numpy as np
import random
import math
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from matplotlib import pylab as plt
################################################################################################################
############################ Global Initialization #############################################################
targetAttribute = 'Poisonous/Edible'
targetValue = []
predictedList= []
# maxGainAttr:  spore-print-color    if  maxInfoGain > 0.144354902043, we get the data classified incorrectly
minimalGainThreshold =0.144
defaultTargetValue = 'p'
rawDataSet = 'MushroomDataSet_Before_PreProcessing.xlsx'
processedDataSet = 'MushroomDataSet_After_PreProcessing.xlsx'
columnToProcess = 'stalk-root'
res = {}
rfTargetValue = []
numberOfTrees = 100
###################################################################################################################
###################################################################################################################
datasetAfterProcessing = pd.read_excel(processedDataSet)
X = datasetAfterProcessing.drop(labels=targetAttribute, axis=1)
y = datasetAfterProcessing[targetAttribute]

for i in range(numberOfTrees):
    random_state = random.randint(0,7)
    X_train_rf,X_test_rf,Y_train_rf,Y_test_rf = train_test_split(X,y,test_size = 0.2 ,random_state = random_state)
    X_train_rf[targetAttribute] = Y_train_rf
    targetValue = initializeList(X_train_rf)
    targetDict = createTree(X_train_rf)
    rfTargetValue.append(targetDict)
for i in list(range(0,len(X_test_rf))):
    predictedVal = predictTarget(targetDict,X_test_rf.iloc[[i]])
    predictedList.append(predictedVal)

printMatrix(Y_test_rf ,predictedList,list(targetValue))
   