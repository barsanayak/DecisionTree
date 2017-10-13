import pandas as pd;
import math;


##################################################
# data class to hold csv data

def entropy(positive, negative):
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


def posNegCount(data):
    grouped_data = data.groupby(['PlayTennis'], as_index=False)

    playTennisGroups = list(grouped_data.groups.keys())
    # print(type(playTennisGroups))
    if (len(playTennisGroups) == 2):
        group_yes = grouped_data.get_group('Yes')
        group_no = grouped_data.get_group('No')
        return len(group_yes), len(group_no)
    elif ((len(playTennisGroups) == 1) & (playTennisGroups[0] == 'Yes')):
        group_yes = grouped_data.get_group('Yes')
        return len(group_yes), 0
    elif ((len(playTennisGroups) == 1) & (playTennisGroups[0] == 'No')):
        group_no = grouped_data.get_group('No')
        return 0, len(group_no)
    else:
        return 0, 0


def calcInforGainForEachAtrribute(df):
    # return maximum info gain
    maxInfoGain =0;
    infoGainDict = {}
    attributes = list(df)
    group_yes, group_no = posNegCount(df)
    totalEntropy = entropy(group_yes, group_no)

    for att in attributes:
        if (att == 'PlayTennis'):
            continue
        grouped_data = df.groupby([att], as_index=False)
        totalRows = len(df)
        subEntropy = 0
        print('keys for ', att, ' total- ', totalRows)

        for key, value in grouped_data:
            eachGroupRows = len(value)
            print('key-', key, ' eachGroupRows- ', eachGroupRows)
            group_yes, group_no = posNegCount(value)
            S = entropy(group_yes, group_no)
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
    df = pd.read_csv("demoData.csv")
    maxGain = calcInforGainForEachAtrribute(df)
    print (maxGain)

if __name__ == "__main__": main()





