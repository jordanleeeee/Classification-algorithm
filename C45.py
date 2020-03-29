import decisionTree as dt
import infoCalculator as cal
import reader
import time


class CFourPointFive(dt.DecisionTree):
    def getGain(self, targetInfo, attribute):
        return (targetInfo - cal.info(self.df, attribute, self.determineAttribute)) / cal.baseInfo(self.df, attribute)

    def getNewNode(self, attribute, value, attributes):
        return CFourPointFive(self.df[self.df[attribute] == value], attributes, self.determineAttribute)


df = reader.readTrainingDataSet('train.txt')
attributes = list(df.keys())
print(df)

start = time.time()

decisionTree = CFourPointFive(df, attributes[:-1], attributes[-1])
print("/////////////////////////")
decisionTree.printTree()
print("/////////////////////////")

testCases = reader.readTestingDataSet('test.txt')
for testCase in testCases:
    predictedClass = decisionTree.predict(testCase)
    print("\n" + attributes[-1] + " of ", end='')
    print(list(testCase.values()))
    print("is classify to ", end='')
    print(predictedClass)

print("time take " + str(time.time() - start))
