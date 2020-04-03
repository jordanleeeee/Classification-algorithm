import reader
import time


class Classifier:
    def __init__(self, df, attributes):
        self.df = df                                    # data frame that store the training data set
        self.attributes = attributes[:-1]               # attributes that used for the prediction
        self.determineAttribute = attributes[-1]        # last attribute is attribute that we want to predict
        self.possibleClasses = df[self.determineAttribute].unique()     # possible class of the prediction
        # A 3 dimensional dictionary where
        # dict[attribute][value][predictedClass] will store the value of P(attribute = value | class = predictedClass)
        self.posteriori = dict()
        # a dictionary that store how likely each class will occur
        self.probiOfEachClass = dict()
        self.__train()

    # train the classifier
    def __train(self):
        for possibleClass in self.possibleClasses:
            self.probiOfEachClass[possibleClass] = len(self.df[self.df[self.determineAttribute] == possibleClass]) / len(self.df)

        for attribute in self.attributes:
            self.posteriori[attribute] = dict()
            uniqueValue = df[attribute].unique()
            for value in uniqueValue:
                probi = dict()
                for possibleClass in self.possibleClasses:
                    partialDf = self.df[self.df[self.determineAttribute] == possibleClass]
                    # use Laplacian correction  (Adding 1 to each case)
                    probi[possibleClass] = \
                        (len(partialDf[partialDf[attribute] == value]) + 1) / (len(partialDf) + len(uniqueValue))
                self.posteriori[attribute][value] = probi

    # return P(c1, c2...cn | determineAttribute = possibleClass) * P(determineAttribute = possibleClass)
    def __calculateConfidence(self, testCase, predictedClass):
        result = 1
        for attribute, value in testCase.items():
            result *= self.posteriori[attribute][value][predictedClass]
        return result * self.probiOfEachClass[predictedClass]

    def naiveBayesPredict(self, testCase):
        predictedClass, maxConfidence = list(), 0
        for possibleClass in self.possibleClasses:
            confidence = self.__calculateConfidence(testCase, possibleClass)
            # choose the class that has highest confidence
            if confidence > maxConfidence:
                maxConfidence = confidence
                predictedClass = list([possibleClass])
            elif confidence == maxConfidence:
                predictedClass.append(possibleClass)
            # print()
            # print(testCase)
            # print(" will belongs to class: " + possibleClass+" has confidence "+str(confidence))
        return predictedClass


df = reader.readTrainingDataSet('train.txt')
attributes = list(df.keys())
# print(df)

start = time.time()

classifier = Classifier(df, attributes)

testCases = reader.readTestingDataSet('test.txt')
for testCase in testCases:
    predictedClass = classifier.naiveBayesPredict(testCase)
    print("\n" + attributes[-1] + " of ", end='')
    print(list(testCase.values()))
    print("is classify to ", end='')
    print(predictedClass)
print("time take " + str((time.time() - start)))
