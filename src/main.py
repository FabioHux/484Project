import numpy as np
from preproc import Preprocessor
from dec_tree import dec_tree
from neur_net import NeuralNetwork
from KNN import KNN
from sklearn.metrics import jaccard_score
import pandas as pd
def kfold(matrix,k,values,clf):
    partition_size=int(np.shape(matrix)[0]/k)
    matrix=matrix.tolist()


    average=0



    for i in range(k):
        part_start=i*partition_size
        part_end=(i+1)*partition_size
        if part_end >=np.shape(matrix)[0]:
            part_end=np.shape(matrix)[0]

    
        part_train = np.append(matrix[:part_start,...], matrix[part_end:,...], axis = 0)
        part_val=matrix[part_start:part_end]

        part_cls_train=np.append(values[:part_start],[values[part_end:]])
        part_cls_val=values[part_start:part_end]
        
        test = Tester(part_cls_val, cls_cnt)

        clf.setUp(part_train,part_cls_train)
        res=clf.predict(part_val)

        print(test.get_jaccard(res))
        print(test.get_f1(res))
        print(test.get_accur(res))
        counter=0
        for x in range(np.size(res,0)):
            if res[x]==part_cls_val[x]:
                counter+=1
                '''
                print(res[x])
                print(part_cls_val[x])
                '''
        average+=counter/partition_size
        
        if counter/partition_size>=.65:
            print(counter/partition_size)

        else:
            print(counter/partition_size)

            
    print("average:"+str(average/k))


    

    

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
    class_count = 50

    values = split_class(preprocessor.getColumn("Lifeexpectancy"), 44, 90, k=5).astype('int')

    '''
    print(np.size(values,0))


    print(preprocessor.getAttributes())
    for x in preprocessor.getMatrix()[0]:
        print(int(x))
    '''
    '''
    print("tree")
    clf=dec_tree(max_depth=5,min_samples_split=3)
    kfold(preprocessor.getMatrix(),5,values,clf)


    clf=NeuralNetwork( activation="tanh",hidden_layer_sizes = (1000,100))
    print("neural net")
    kfold(preprocessor.getMatrix(),10,values,clf)'''
    
    clf=KNN()
    print("KNN")
    kfold(preprocessor.getMatrix(),10,values,clf)

    '''
    clf=NeuralNetwork(solver="adam", activation="relu", hidden_layer_sizes = (200,25,200,25,100))
    kfold(preprocessor.getMatrix(),4,values,clf, class_count)'''
    
    kfold(preprocessor.getMatrix(),4,values,clf, class_count)


main()
