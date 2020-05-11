from sklearn import tree
import numpy as np

class dec_tree():
    def __init__(self,criterion='gini',splitter='best',max_depth=None,min_samples_split=2,min_samples_leaf=1):
        self.clf=tree.DecisionTreeClassifier(criterion,splitter,max_depth,min_samples_split,min_samples_leaf)

    def setUp(self,matrix,values):
        self.clf.fit(matrix, values)

    def predict(self, matrix):
        return self.clf.predict(matrix)

