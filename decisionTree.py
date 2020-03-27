import numpy as np
import pandas as pd
import infoCalculator as cal


class decisionTree():
    def __init__(self, df, attributes, determineAttribute):
        print("\n\nhello")
        print(attributes)
        self.attribute = None
        self.subNode = dict()
        self.df = df
        self.determineAttribute = determineAttribute
        self.attributes = attributes
        self.mineTree()

    def shouldStop(self):
        # All samples for a given node belong to the same class
        if len(self.df[self.determineAttribute].unique()) == 1:
            print("no need to split")
            return True
        # There are no remaining attributes for further partitioning
        if len(self.attributes) == 0:
            return True
        # There are no samples left
        if len(self.df) == 0:
            return True
        return False

    def mineTree(self):
        if self.shouldStop():
            return

        # when the tree can be split and need to split
        targetInfo = cal.baseInfo(self.df, self.determineAttribute)
        splitAttribute, maxGain = None, 0
        for attribute in self.attributes:
            # gain = (targetInfo - info(df, attribute, attributes[-1]))/baseInfo(df, attribute)
            gain = (targetInfo - cal.info(self.df, attribute, self.determineAttribute))
            if gain > maxGain:
                splitAttribute = attribute
                maxGain = gain
            print("attribute: " + attribute + " has gain = " + str(gain))
        print("split at " + splitAttribute)
        self.splitTree(splitAttribute)

    def splitTree(self, attribute):
        self.attribute = attribute
        keys = self.df[attribute].unique()  # unique value
        for value in keys:
            temp = self.attributes.copy()
            temp.remove(attribute)
            self.subNode[value] = decisionTree(self.df[self.df[attribute] == value], temp, self.determineAttribute)


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


attributes, records = readTrainingDataSet('smallTrain.txt')
df = pd.DataFrame(np.array(records), columns=attributes)
print(df)

# targetInfo = cal.baseInfo(df, attributes[-1])
# print(attributes[:-1])
# print(targetInfo)

decisionTree = decisionTree(df, attributes[:-1], attributes[-1])
