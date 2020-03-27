import numpy as np
import pandas as pd

class decisionTree():
    def __init__(self, df):
        self.attribute = None
        self.subNode = list()
        self.df = df

    def splitTree(self, attribute):
        self.attribute = attribute
        keys = self.df[attribute].unique()  # unique value
        for value in keys:
            self.subNode.append(self.df[self.df[attribute] == value])

    def addTreeNode(self, node):
        self.subNode.append(node)
