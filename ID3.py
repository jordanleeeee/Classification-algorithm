import decisionTree as dt
import infoCalculator as cal
import reader
import time


class ID3(dt.DecisionTree):
    def getGain(self, targetInfo, attribute):
        return targetInfo - cal.info(self.df, attribute, self.determineAttribute)

    def getNewNode(self, attribute, value, attributes):
        return ID3(self.df[self.df[attribute] == value], attributes, self.determineAttribute)


df = reader.readTrainingDataSet('train.txt')
attributes = list(df.keys())
# print(df.shape[0])
# print(len(df.drop_duplicates()))

start = time.time()
decisionTree = ID3(df, attributes[:-1], attributes[-1])
# print("/////////////////////////")
# decisionTree.printTree()
# print("/////////////////////////")
# print(decisionTree.getNumOfLeaf())

testCases = reader.readTestingDataSet('test.txt')
for testCase in testCases:
    predictedClass = decisionTree.predict(testCase)
    print("\n" + attributes[-1] + " of ", end='')
    print(list(testCase.values()))
    print("is classify to ", end='')
    print(predictedClass)

print("time take " + str(time.time() - start))
