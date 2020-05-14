import numpy as np
from copy import copy
from sklearn.metrics.pairwise import cosine_similarity

class KNN():
    def __init__(self, cls_cnt, k=10):
        self.cls_cnt = cls_cnt
        self.k = k

    def setUp(self,matrix,values):
        self.matrix=matrix
        self.values=values
    
    def setClsCnt(self, cls_cnt):
        self.cls_cnt = cls_cnt

    def predict(self, matrix):
        sim=cosine_similarity(matrix,self.matrix)

        ret_val = np.empty(0)

        multi_var = np.zeros(self.cls_cnt)

        for x in sim:
            multi_var *= 0

            endval = copy(self.values)
            newk = copy(x)
            mapped = sorted(zip(newk,endval), reverse = True)

            for y in range(self.k):
                multi_var[mapped[y][1]] += 1
            
            abs=0
            i = 0
            hi = 0
            for y in multi_var:
                if y > abs:
                    abs = y
                    hi = i
                i += 1
            ret_val = np.append(ret_val, hi)
        
        return ret_val