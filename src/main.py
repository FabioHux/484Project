import numpy as np
from preproc import Preprocessor
from sklearn.metrics import jaccard_score
import pandas as pd

def split_class(values, low, high, k = 2):
    if low > np.amin(values) or high < np.amax(values):
        print(str(np.amin(values)) + " " + str(np.amax(values)))
        return None
    new_values = np.empty(0)
    step = (high - low) / k

    for x in values:
        val = (x - low) // step
        if val == k:
            val -= 1
        new_values = np.append(new_values, val)
    
    return new_values


def main():
    f = open("../doc/led.csv", "r")

    preprocessor = Preprocessor()
    preprocessor.preprocess(f)
    preprocessor.cleanUnfilled()

    values = split_class(preprocessor.getColumn("Lifeexpectancy"), 44, 90, k=4)

    
    print(preprocessor.getAttributes())

main()
