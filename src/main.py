import numpy as np
from preproc import Preprocessor
from dec_tree import dec_tree
from neur_net import NeuralNetwork
from sklearn.metrics import jaccard_score
import pandas as pd
def kfold(matrix,k,values,clf):
    partition_size=int(np.shape(matrix)[0]/k)
    matrix=matrix.tolist()
    for i in range(k):
        part_start=i*partition_size
        part_end=(i+1)*partition_size
        if part_end >=np.shape(matrix)[0]:
            part_end=np.shape(matrix)[0]

        '''matrix[...,0:part_start],matrix[...,part_end:np.shape(matrix)[0]]'''
    
        part_train=matrix[:part_start]
        for x in matrix[part_end:]:
            part_train.append(x)

        #print(matrix[...,0:part_start])
        part_val=matrix[part_start:part_end]
        for x in part_val:
            if x in part_train:
                print(True)

        part_cls_train=np.append(values[:part_start],[values[part_end:]])
        part_cls_val=values[part_start:part_end]

        #print(part_cls_train)
        clf.setUp(part_train,part_cls_train)
        res=clf.predict(part_val)
        counter=0
        for x in range(np.size(res,0)):
            if res[x]==part_cls_val[x]:
                counter+=1
                '''
                print(res[x])
                print(part_cls_val[x])
                '''
        print(counter)









    

    

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

    '''
    print(np.size(values,0))
    print(preprocessor.getAttributes())
    for x in preprocessor.getMatrix()[0]:
        print(int(x))
    '''

    clf=dec_tree()
    kfold(preprocessor.getMatrix(),3,values,clf)
    
    clf=NeuralNetwork()
    kfold(preprocessor.getMatrix(),3,values,clf)
    
    



main()
