import numpy as np
from preproc import Preprocessor
from dec_tree import dec_tree
from neur_net import NeuralNetwork
from testing import Tester

def kfold(matrix,k,values,clf, cls_cnt):
    partition_size=int(np.shape(matrix)[0]/k)


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
        print(counter)
        print("")




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

    values = split_class(preprocessor.getColumn("Lifeexpectancy"), 40, 90, k=class_count)

    clf=dec_tree()
    kfold(preprocessor.getMatrix(),4,values,clf, class_count)

    clf=NeuralNetwork(solver="adam", activation="relu", hidden_layer_sizes = (200,25,200,25,100))
    kfold(preprocessor.getMatrix(),4,values,clf, class_count)


main()
