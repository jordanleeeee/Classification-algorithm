import numpy as np


def sum(*num):
    result = 0
    for i in num:
        result += i
    return result


print(sum(1,2,3,4,3))


