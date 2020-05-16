import numpy as np
from preproc import Preprocessor
from dec_tree import dec_tree
from neur_net import NeuralNetwork
from KNN import KNN
from testing import Tester
from matplotlib import pyplot as plt

"""
Kfold cross validation

Input: Entries in the form of a matrix, k-fold count, list of expected values, type of classifier, counts of classes
Requirements:
>>>Number of entries equal number of expected values
>>>k-fold is at least 2
>>>Classifier must be already initialized, must have setUp(matrix, values) and predict(matrix) function ready
>>>list of values must have already been cleaned using split_class

Return: Tuple containing a matrix of scores for each fold, and a weighted score for all the folds
>>>Tests follow in this order evaluation: [jaccard, f1, accuracy, dist_acc]
>>>See testing.py for more details  
"""
def kfold(matrix,k,values,clf, cls_cnt):
    kvalues = np.empty((4,0))
    partition_size=int(np.shape(matrix)[0]/k)
    kweights = np.ones(k) * (1/k)

    for i in range(k):
        part_start=i*partition_size
        part_end=(i+1)*partition_size
        if part_end >=np.shape(matrix)[0]:
            part_end=np.shape(matrix)[0]

        part_train = np.append(matrix[:part_start,...], matrix[part_end:,...], axis = 0)
        part_val=matrix[part_start:part_end]

        part_cls_train=np.append(values[:part_start],[values[part_end:]])
        part_cls_val=values[part_start:part_end]
        
        test = Tester(part_cls_val, cls_cnt, 50/cls_cnt)

        clf.setUp(part_train,part_cls_train)
        res=clf.predict(part_val)

        kvalues = np.append(kvalues, [[test.get_jaccard(res)],[test.get_f1(res)],[test.get_accur(res)], [test.get_dist_acc(res)]], axis = 1)
    
    scores = np.matmul(kvalues, kweights)
    return (kvalues, scores)

"""
Country fold cross validation

Will evaluate the models based on each countries performance
Input: Entries in the form of a matrix, k-fold count, list of expected values, type of classifier, counts of classes, list of countries
Requirements:
>>>Number of entries equal number of expected values
>>>Number of countries equal number of expected values
>>>Countries must be in lexigraphical order
>>>k-fold is at least 2
>>>Classifier must be already initialized, must have setUp(matrix, values) and predict(matrix) function ready
>>>list of values must have already been cleaned using split_class

Return: Tuple containing a matrix of scores for each country fold, and a weighted score for all the folds
>>>Tests follow in this order evaluation: [jaccard, f1, accuracy, dist_acc]
>>>See testing.py for more details  
"""
def countryfold(matrix, k, values, clf, cls_cnt, countries):
    i = 0
    kvalues = np.empty((3,0))
    kweights = np.empty(0)
    len_list = np.shape(countries)[0]

    while i < np.shape(countries)[0]:
        country = countries[i]
        c_i = np.where(countries == country)
        part_start = c_i[0][0]
        part_end = c_i[0][np.shape(c_i)[1] - 1]
        kweights = np.append(kweights, np.shape(c_i)[1] / len_list)

        part_train = np.append(matrix[:part_start,...], matrix[part_end + 1:,...], axis = 0)
        part_val=matrix[part_start:part_end + 1]

        part_cls_train=np.append(values[:part_start],[values[part_end + 1:]])
        part_cls_val=values[part_start:part_end + 1]
        test = Tester(part_cls_val, cls_cnt, 50/cls_cnt)
        clf.setUp(part_train,part_cls_train)
        res=clf.predict(part_val)
        i = part_end + 1

        kvalues = np.append(kvalues, [[test.get_jaccard(res)],[test.get_f1(res)],[test.get_accur(res)], [test.get_dist_acc(res)]], axis = 1)

    scores = np.matmul(kvalues, kweights)
    return (kvalues, scores)

"""
List reformatter for transforming continuous data into multi classes

Input: List of continuous values, lowest value of range, highest value of range, number of classes to split into
Requirements:
>>>Low must be <= the lowest value in values
>>>high must be >= the highest value in values
>>>k must be at minimum 1

Return: New list of the same size of values, with the relative continuous values transformed into classes
>>>None will be returned on unacceptable low and high

Note: the classes will be divided evenly so for a range of 0:10 with 5 divisions, each partition will be of size 2.0
"""
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

"""
Function to enact first expirement.

Experiment concerns with seeing the performance of the three models (KNN, Neural Network, Decision Tree) based on the modification of the class count in the data

The method of testing will be through the kfold function (see kfold()) with 10 folds. The experiment will only take the f1-score as a determination.

It will return a matrix (2D numpy array) of every model's f1 score as a row and n columns representing a different class count that it was tested with (Note the class count go from [4,24] in increments of 2)
"""
def experiment1(preprocessor):
    results = np.empty((0,3))
    values = preprocessor.getColumn("Lifeexpectancy")
    clf = [KNN(0, k = 20), NeuralNetwork(activation = "tanh", hidden_layer_sizes = (100,25)), dec_tree()]
    for cls_cnt in range(4,25,2):
        clf[0].setClsCnt(cls_cnt)
        cls_values = split_class(values, 40, 90, k=cls_cnt).astype('int')
        results = np.append(results, [[kfold(preprocessor.getMatrix(), 10, cls_values, x, cls_cnt)[1][1] for x in clf]], axis = 0)
    
    return results

"""
Function to enact the second experiment

Experiment concerns with seeing the performance of the three models on measuring each country.

Function will test each model with countryfold (see countryfold) to see its score and how it performs.

It will return a matrix (2D numpy array) of every model's score (in this instance jaccard, f1-score, and accuracy) as the column, with each row representing the model
"""
def experiment2(preprocessor):
    results = np.empty((0,3))
    values = split_class(preprocessor.getColumn("Lifeexpectancy"), 40, 90, k=10).astype('int')
    for clf in [KNN(10, k = 20), NeuralNetwork(activation = "tanh", hidden_layer_sizes = (100,25)), dec_tree()]:
        results = np.append(results, [countryfold(preprocessor.getMatrix(), 10, values, clf, 10,preprocessor.getCountries())[1]], axis = 0)
    return results

"""
Main function of the program, takes in a filename (as a string) of the csv file required to extract.

Modifiable to user's need
"""
def main(filename):
    f = open(filename, "r")
    
    preprocessor = Preprocessor()
    preprocessor.preprocess(f)
    preprocessor.cleanUnfilled()
    print(experiment1(preprocessor))


main("../doc/led.csv")
