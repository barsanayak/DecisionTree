# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 19:24:21 2017

@author: nithi
"""
import numpy as np
import pandas as pd
import math

def entropyCalculation(positive,negative):
    total = positive + negative
    positiveTemp = positive / total
    negativeTemp = negative / total

    positiveEntropy = 0
    negativeEntropy = 0

    if (positiveTemp != 0):
        positiveEntropy = positiveTemp * math.log2(positiveTemp)

    if (negativeTemp != 0):
        negativeEntropy = negativeTemp * math.log2(negativeTemp)

    entropy = 0 - (positiveEntropy + negativeEntropy)
    return entropy


def posNegCount(training_data):
    grouped_data = training_data.groupby(['Poisonous/Edible'], as_index=False)
    PoisonousEdibleGroups = list(grouped_data.groups.keys())
    if (len(PoisonousEdibleGroups) == 2):
        group_edible = grouped_data.get_group('e')
        group_poisonous = grouped_data.get_group('p')
        return len(group_edible), len(group_poisonous)
    elif ((len(PoisonousEdibleGroups) == 1) & (PoisonousEdibleGroups[0] == 'e')):
        group_edible = grouped_data.get_group('e')
        return len(group_edible), 0
    elif ((len(PoisonousEdibleGroups) == 1) & (PoisonousEdibleGroups[0] == 'p')):
        group_poisonous = grouped_data.get_group('p')
        return 0, len(group_poisonous)
    else:
        return 0, 0
    
def calcInforGainForEachAtrribute(df):
    # return maximum info gain
    maxInfoGain =0;
    infoGainDict = {}
    attributes = list(df)
    group_yes, group_no = posNegCount(df)
    totalEntropy = entropyCalculation(group_yes, group_no)

    for att in attributes:
        if (att == 'Poisonous/Edible'):
            continue
        grouped_data = df.groupby([att], as_index=False)
        totalRows = len(df)
        subEntropy = 0
        print('keys for ', att, ' total- ', totalRows)

        for key, value in grouped_data:
            eachGroupRows = len(value)
            print('key-', key, ' eachGroupRows- ', eachGroupRows)
            group_yes, group_no = posNegCount(value)
            S = entropyCalculation(group_yes, group_no)
            valEntrpy = eachGroupRows / totalRows
            subEntropy = subEntropy + valEntrpy * S

        individualEntrpyGain = totalEntropy - subEntropy
        infoGainDict[att] = individualEntrpyGain
        print('gain-', att, ' - ', individualEntrpyGain)
        if (individualEntrpyGain > maxInfoGain):
            maxInfoGain = individualEntrpyGain
            maxGainAttr = att;
    return maxGainAttr
    

def main():
   # df = pd.read_csv("demoData.csv")
   datasetforAssignment = pd.read_excel('MushroomDataSet_Before_PreProcessing.xlsx')
   x = datasetforAssignment.iloc[:, 11].values
   df1 = datasetforAssignment.fillna(datasetforAssignment['stalk-root'].value_counts().index[0])
   writer = pd.ExcelWriter('MushroomDataSet_After_PreProcessing.xlsx', engine='xlsxwriter')
   df1.to_excel(writer, sheet_name='Sheet1')
   writer.save()

   datasetAfterProcessing = pd.read_excel('MushroomDataSet_After_PreProcessing.xlsx')
   split_value = (int) (len(datasetAfterProcessing.index)*0.8)
   training_data = pd.DataFrame(datasetAfterProcessing.iloc[:split_value,:].values)
   training_data.columns= ['Poisonous/Edible','cap-shape	','cap-surface','cap-color','bruises?','odor','gill-attachment','gill-spacing','gill-size','gill-color','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring	','stalk-color-above-ring','stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type','spore-print-color',	'population','habitat']
 
   maxGain = calcInforGainForEachAtrribute(training_data)
   print (maxGain)

if __name__ == "__main__": main()