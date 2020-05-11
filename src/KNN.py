import numpy as np
from copy import copy
from sklearn.metrics.pairwise import cosine_similarity

class KNN():
    def __init__(self):
        sim=0

    def setUp(self,matrix,values):
        self.matrix=matrix
        self.values=values

    def predict(self, matrix,k=10):
        sim=cosine_similarity(matrix,self.matrix)
        low=1000
        high=0
        for x in self.values:
            if x<low:
                low=x
            if x >high:
                high=x


        ret_val=[]

        for x in sim:
            
            multi_var=[0 for x in range(high-low+1)]


            num=0
            endval=copy(self.values)
            newk=copy(x)
            mapped = zip(newk,endval)
            mapped = sorted(mapped, reverse = True)

            for y in range(0,k):
                multi_var[mapped[y][1]-1]+=1
            
            abs=0
            for y in multi_var:
                if y>abs:
                    abs=y

            for y in range(len(multi_var)):
                if multi_var[y]==abs:
                    ret_val.append(y+1)
                    break
        return ret_val