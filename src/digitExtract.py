from preproc import Preprocessor
import numpy as np

"""
File made by Vincent to initially clean the original excel file
"""
#seperates by tabs and lines and writes to another file.
file_t=open("../doc/led.csv",'r')
file_w=open("led2.csv",'w')



preprocessor = Preprocessor()
preprocessor.preprocess(file_t)
matrix = preprocessor.getMatrix()
print(np.shape(matrix))
preprocessor.cleanUnfilled()
matrix = preprocessor.getMatrix().tolist()
print(np.shape(matrix))

    
w=preprocessor.getAttributes().tolist()
w.insert(0,"Years")
w.insert(0,"Countries")


for x in range(len(w)):
        if x==len(w)-1:
            file_w.write(str(w[x]))
        else:
            file_w.write(str(w[x])+",")

w=matrix
countries= preprocessor.getCountries()
for x in range(len(w)):
    file_w.write(countries[x]+",")
    file_w.write(str(x)+",")
    for y in range(len(w[x])):

        if y==len(w[x])-1:
            file_w.write(str(w[x][y]))
        else:
            file_w.write(str(w[x][y])+",")
    file_w.write("\n")