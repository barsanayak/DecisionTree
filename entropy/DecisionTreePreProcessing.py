# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 19:24:21 2017

@author: nithi
"""
import numpy as np
import pandas as pd

datasetforAssignment = pd.read_excel('MushroomDataSet_Before_PreProcessing.xlsx')
x = datasetforAssignment.iloc[:, 11].values
df1 = datasetforAssignment.fillna(datasetforAssignment['stalk-root'].value_counts().index[0])
writer = pd.ExcelWriter('MushroomDataSet_After_PreProcessing.xlsx', engine='xlsxwriter')
df1.to_excel(writer, sheet_name='Sheet1')
writer.save()

datasetAfterProcessing = pd.read_excel('MushroomDataSet_After_PreProcessing.xlsx')
x_splitting = datasetAfterProcessing.iloc[:, 1:].values
y_splitting = datasetAfterProcessing.iloc[:, 0].values

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x_splitting,y_splitting,test_size= 0.2,random_state = 42)

