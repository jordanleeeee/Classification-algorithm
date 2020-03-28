import decisionTree
import infoCalculator as cal
import reader
import numpy as np
import pandas as pd


class CFourPointFive(decisionTree.DecisionTree):
    def getGain(self, targetInfo, attribute):
        return (targetInfo - cal.info(self.df, attribute, self.determineAttribute)) / cal.baseInfo(self.df, attribute)

    def getNewNode(self, attribute, value, attributes):
        return CFourPointFive(self.df[self.df[attribute] == value], attributes, self.determineAttribute)


attributes, records = reader.readTrainingDataSet('smallTrain.txt')
df = pd.DataFrame(np.array(records), columns=attributes)
print(df)

decisionTree = CFourPointFive(df, attributes[:-1], attributes[-1])
decisionTree.printTree()

testCases = reader.readTestingDataSet('guess.txt')
for testCase in testCases:
    predictedClass = decisionTree.predict(testCase)
    print("\n" + attributes[-1] + " of ", end='')
    print(testCase)
    print("is classify to ", end='')
    print(predictedClass)
