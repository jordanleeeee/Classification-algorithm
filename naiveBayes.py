import reader
import pandas as pd
import numpy as np


# P(c1, c2...cn | determineAttribute = possibleClass) * P(determineAttribute = possibleClass)
def calculateConfidence(df, testCase, possibleClass, determineAttribute):
    partialDf = df[df[determineAttribute] == possibleClass]  # df that contain only: determineAttribute = possibleClass
    # P(determineAttribute = possibleClass)
    probiOfSuchClass = len(partialDf) / len(df)
    # P(c1, c2...cn | determineAttribute = possibleClass)
    conditionalProbi = 1
    for attribute, value in testCase.items():
        # P(attribute = value | determineAttribute = possibleClass)
        temp = len(partialDf[partialDf[attribute] == value]) / len(partialDf)
        conditionalProbi *= temp
    return conditionalProbi * probiOfSuchClass


def naiveBayesPredict(df, testCase, determineAttribute):
    possibleClasses = df[determineAttribute].unique()
    predictedClass, maxConfidence = None, 0
    print()
    for possibleClass in possibleClasses:
        confidence = calculateConfidence(df, testCase, possibleClass, determineAttribute)
        print(testCase, end='')
        print(" belong to " + possibleClass + " has confidence " + str(confidence))
        if confidence > maxConfidence:
            maxConfidence = confidence
            predictedClass = possibleClass
    return predictedClass


attributes, records = reader.readTrainingDataSet('smallTrain2.txt')
df = pd.DataFrame(np.array(records), columns=attributes)
print(df)

testCases = reader.readTestingDataSet('guess2.txt')
for testCase in testCases:
    predictedClass = naiveBayesPredict(df, testCase, attributes[-1])
    print(attributes[-1] + " of ", end='')
    print(testCase)
    print("is classify to ", end='')
    print(predictedClass)
