import infoCalculator as cal


class DecisionTree:
    def __init__(self, df, attributes, determineAttribute):
        # print("\n\nhello")
        # print(attributes)
        self.df = df
        self.attributes = attributes
        self.determineAttribute = determineAttribute  # the class that the tree will predict

        self.attribute = None
        self.subNode = dict()
        self.classification = None

        self.mineTree()

    def predict(self, testCase):
        if self.attribute is not None:
            return self.subNode[testCase[self.attribute]].predict(testCase)     # recursively go though decision node
        else:
            return self.classification

    def getDepth(self):
        if len(self.subNode) == 0:
            return 0
        else:
            maxDepth = -1
            for subTree in self.subNode.values():
                temp = subTree.getDepth()
                if temp > maxDepth:
                    maxDepth = temp
            return 1+maxDepth

    def getNumOfLeaf(self):
        if len(self.subNode) == 0:
            return 1
        sum = 0
        for subTree in self.subNode.values():
            sum += subTree.getNumOfLeaf()
        return sum

    def printTree(self):
        print("\ntree node")
        print(self.attribute)
        print(self.subNode.keys())
        for node in self.subNode.values():
            node.printTree()
        if self.classification is not None:
            print("classification is")
            print(self.classification)

    def shouldStop(self):
        # There are no samples left
        if len(self.df) == 0:
            return True
        # All samples for a given node belong to the same class
        if len(self.df[self.determineAttribute].unique()) == 1:
            return True
        # There are no remaining attributes for further partitioning
        if len(self.attributes) == 0:
            return True
        return False

    def getGain(self, targetInfo, attribute):
        pass

    def mineTree(self):
        if self.shouldStop():
            possibleClass = self.df[self.determineAttribute].unique()
            self.classification = list(possibleClass)
            return

        # when the tree can be split and need to split
        targetInfo = cal.baseInfo(self.df, self.determineAttribute)
        splitAttribute, maxGain = None, 0
        for attribute in self.attributes:
            gain = self.getGain(targetInfo, attribute)
            # don't give me warning, so annoying
            # noinspection PyTypeChecker
            if gain > maxGain:
                splitAttribute = attribute
                maxGain = gain
        #     print("attribute: " + attribute + " has gain = " + str(gain))
        # print("split at " + splitAttribute)
        self.splitTree(splitAttribute)

    def getNewNode(self, attribute, value, temp):
        pass

    def splitTree(self, attribute):
        self.attribute = attribute
        newAttributes = self.attributes.copy()
        newAttributes.remove(attribute)
        keys = self.df[attribute].unique()  # unique value
        for value in keys:
            self.subNode[value] = self.getNewNode(attribute, value, newAttributes)
