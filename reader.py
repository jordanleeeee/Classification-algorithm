import numpy as np
import pandas as pd


def decomposeLine(line):
    line = line[:-1]            # remove \n in the char sequence
    return line.split(",")


def readTestingDataSet(path):
    database = open(path, 'r')
    line = database.readline()
    attributes = decomposeLine(line)

    records = list()
    line = database.readline()
    while line != "":
        record = decomposeLine(line)
        temp = dict()
        for i in range(len(attributes)):
            temp[attributes[i]] = record[i]
        records.append(temp)
        line = database.readline()
    return records


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
    return pd.DataFrame(np.array(records), columns=attributes)


def readTrainingDataSet2(path):
    database = open(path, 'r')
    line = database.readline()
    line = database.readline()
    trainX = list()
    trainY = list()
    while line != "":
        record = decomposeLine(line)
        trainX.append(list(record[:-1]))
        trainY.append(record[-1])
        line = database.readline()
    return trainX, trainY