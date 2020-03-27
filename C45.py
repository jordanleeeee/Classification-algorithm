import decisionTree
import infoCalculator as cal
import reader
import numpy as np
import pandas as pd


class CFourPointFive(decisionTree.DecisionTree):
    def __init__(self, df, attributes, determineAttribute):
        decisionTree.DecisionTree.__init__(self, df, attributes, determineAttribute)

    def getGain(self, targetInfo, attribute):
        return (targetInfo - cal.info(self.df, attribute, self.determineAttribute))/cal.baseInfo(self.df, attribute)

    def getNewNode(self, attribute, value, temp):
        return CFourPointFive(self.df[self.df[attribute] == value], temp, self.determineAttribute)


attributes, records = reader.readTrainingDataSet('smallTrain.txt')
df = pd.DataFrame(np.array(records), columns=attributes)
print(df)

decisionTree = CFourPointFive(df, attributes[:-1], attributes[-1])
decisionTree.printTree()

testCases = reader.readTestingDataSet('guess.txt')
for testCase in testCases:
    predictedClass = decisionTree.predict(testCase)
    print("\n" + attributes[-1] + " of")
    print(testCase)
    print("is predicted to belong to ")
    print(predictedClass)