import numpy as np
import pandas as pd
import decisionTree as tree


def decomposeLine(line):
    line = line[:-1]            # remove \n in the char sequence
    return line.split(",")


def readTrainingDataSet(path):
    database = open(path, 'r')
    line = database.readline()
    attributes = decomposeLine(line)

    records = list()
    line = database.readline()
    while line != "":
        record = decomposeLine(line)
        records.append(record)
        line = database.readline()
    return attributes, records


def infoCalculation(num):
    total = 0
    for i in num:
        total += i
    result = 0
    for i in num:
        if i == 0:
            return 0
        temp = i/total
        result -= temp * np.log2(temp)
    return result


def baseInfo(df, determinAttribute):
    keys = df[determinAttribute].unique()  # last column unique value
    numOfData = len(df)
    nums = list()
    for value in keys:
        nums.append(len(df[df[determinAttribute] == value]))
    return infoCalculation(nums)


def info(df, attribute, determinAttribute):
    numOfData = len(df)
    info = 0
    keys = df[attribute].unique()  # unique value
    for value in keys:
        occurances = len(df[df[attribute] == value])
        probOfOccur = occurances/numOfData
        info += probOfOccur * baseInfo(df[df[attribute] == value], determinAttribute)
    return info


attributes, records = readTrainingDataSet('smallTrain.txt')
df = pd.DataFrame(np.array(records), columns=attributes)
print(df)

targetInfo = baseInfo(df, attributes[-1])
print(targetInfo)

decisionTree = tree.decisionTree(df)
while True:
    splitAttribute, maxGain = None, 0
    for attribute in attributes[:-1]:
        # gain = (targetInfo - info(df, attribute, attributes[-1]))/baseInfo(df, attribute)
        gain = (targetInfo - info(df, attribute, attributes[-1]))
        if gain > maxGain:
            splitAttribute = attribute
            maxGain = gain
        print("attribute: "+attribute+" has gain = "+str(gain))
    print("split at " + splitAttribute)


# incompleteForm = data[data['form'] == 'incomplete']
# incompleteForm = incompleteForm[incompleteForm['health'] == 'recommended']
# print(incompleteForm)
# print(len(incompleteForm))
