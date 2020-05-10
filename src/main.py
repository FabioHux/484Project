import numpy as np
from preproc import Preprocessor

f = open("../doc/led.csv", "r")

preprocessor = Preprocessor()
preprocessor.preprocess(f)
matrix = preprocessor.getMatrix()
print(np.shape(matrix))

preprocessor.cleanUnfilled()
matrix = preprocessor.getMatrix()
print(np.shape(matrix))
print(preprocessor.getAttributes())