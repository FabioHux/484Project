from sklearn import tree
import numpy as np

from sklearn import tree

class dec_tree():
    def __init__(self):
        self.clf=tree.DecisionTreeClassifier()

    def setUp(self,matrix,values):
        self.clf.fit(matrix, values)

    def predict(self, matrix):
        return self.clf.predict(matrix)

