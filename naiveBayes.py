import reader
import time


def calculateConfidence(testCase, conditionalProbi, predictedClass, probiOfEachClass):
    result = 1
    for attribute, value in testCase.items():
        result *= conditionalProbi[attribute][value][predictedClass]
    return result * probiOfEachClass[predictedClass]


def naiveBayesPredict(testCase, conditionalProbi, possibleClasses, probiOfEachClass):
    predictedClass, maxConfidence = list(), 0
    for possibleClass in possibleClasses:
        confidence = calculateConfidence(testCase, conditionalProbi, possibleClass, probiOfEachClass)
        if confidence > maxConfidence:
            maxConfidence = confidence
            predictedClass = list([possibleClass])
        elif confidence == maxConfidence:
            predictedClass.append(possibleClass)
        # print(testCase, end='')
        # print(" will belongs to class: "+ possibleClass+" has confidence"+str(confidence))
    return predictedClass


# will return a m dimensional dictionary where
# conditionalProbi[attribute][value][predictedClass] will store the value of P(attribute = value | predictedClass)
def determineConditionalProbi(df, attributes, determineAttribute):
    possibleClasses = df[determineAttribute].unique()
    conditionalProbi = dict()
    for attribute in attributes:
        conditionalProbi[attribute] = dict()
        for value in df[attribute].unique():
            probi = dict()
            for possibleClass in possibleClasses:
                partialDf = df[df[determineAttribute] == possibleClass]
                probi[possibleClass] = len(partialDf[partialDf[attribute] == value]) / len(partialDf)
            conditionalProbi[attribute][value] = probi
    return conditionalProbi


def getProbiOfEachClass(df):
    probiOfEachClass = dict()
    for aClass in df.unique():
        probiOfEachClass[aClass] = len(df[df == aClass])/len(df)
    return probiOfEachClass


df = reader.readTrainingDataSet('train.txt')
attributes = list(df.keys())
print(df)
conditionalProbi = determineConditionalProbi(df, attributes[:-1], attributes[-1])
probiOfEachClass = getProbiOfEachClass(df[attributes[-1]])

start = time.time()
testCases = reader.readTestingDataSet('test.txt')
for testCase in testCases:
    predictedClass = naiveBayesPredict(testCase, conditionalProbi, df[attributes[-1]].unique(), probiOfEachClass)
    print("\n"+attributes[-1] + " of ", end='')
    print(list(testCase.values()))
    print("is classify to ", end='')
    print(predictedClass)
print("time take " + str((time.time() - start)*1000))
