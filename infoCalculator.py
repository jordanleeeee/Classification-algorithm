import numpy as np


def entropy(num):
    total = sum(num)
    result = 0
    for i in num:
        if i == 0:
            continue
        temp = i/total
        result -= temp * np.log2(temp)
    return result


def baseInfo(df, determineAttribute):
    keys = df[determineAttribute].unique()  # column unique value
    nums = list()
    for value in keys:
        nums.append(len(df[df[determineAttribute] == value]))
    return entropy(nums)


def info(df, attribute, determineAttribute):
    numOfData = len(df)
    result = 0
    keys = df[attribute].unique()  # unique value
    for value in keys:
        occurrence = len(df[df[attribute] == value])
        probOfOccur = occurrence/numOfData
        result += probOfOccur * baseInfo(df[df[attribute] == value], determineAttribute)
    return result
