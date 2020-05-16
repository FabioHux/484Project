import numpy as np
import math

class Tester():

    """
    Constuctor of Tester

    Input: Takes in a list (1D array) of expected values, k-number of class counts, size of steps of each class
    """
    def __init__(self, expected, k, step_size):
        self.k = k
        self.expected = expected
        self.matrix, self.weights = self.transform(expected)
        self.onesmat = np.ones(np.shape(self.matrix))
        self.step_size = step_size
        self.exp_dist = self.eucl_dist(expected * step_size)

    """
    Function used to find the size of a given vector
    """
    def eucl_dist(self, values):
        return math.sqrt(np.sum(np.power(values,2)))

    """
    Function to create a matrix of k*n where k is the number of classes and n is the number of values in the 1D array
    The matrix will contain k rows, each x row being the specific class of the data. Each row is a binary vector representing the values do and don't contain the x class
    Ex.
    >>>transform([0,2,1])
    >>> matrix:
        [[1,0,0],
        [0,0,1],
        [0,1,0]]

        weights:
        [0.333333,0.333333,0.333333]
    
    Note: The return will by a (k,n) numpy matrixs
    """
    def transform(self, values):
        matrix = np.empty((0, np.shape(values)[0]), dtype='int')
        weights = np.empty(0)

        for i in range(self.k):
            row = (values == i) * 1
            matrix = np.append(matrix, [row], axis = 0)
            weights = np.append(weights, np.sum(row) / np.shape(values)[0])

        return matrix, weights

    """
    Function to return the number of true positives that a given matrix has with the expected values matrix
    Note: mat should be the same size as the given matrix
    """
    def get_tp(self, mat):
        return np.matmul((self.matrix * mat), np.ones(np.shape(mat)[1]))
    
    """
    Function to return the number of true negatives that a given matrix has with the expected values matrix
    Note: mat should be the same size as the given matrix
    """
    def get_tn(self, mat):
        return np.matmul(((self.matrix ^ 1) * (mat ^ 1)), np.ones(np.shape(mat)[1]))

    """
    Function to return the number of false positives that a given matrix has with the expected values matrix
    Note: mat should be the same size as the given matrix
    """
    def get_fp(self, mat):
        return np.matmul(((self.matrix ^ 1) * mat), np.ones(np.shape(mat)[1]))

    """
    Function to return the number of false negatives that a given matrix has with the expected values matrix
    Note: mat should be the same size as the given matrix
    """
    def get_fn(self, mat):
        return np.matmul((self.matrix * (mat ^ 1)), np.ones(np.shape(mat)[1]))

    """
    Function to determine the weighted accuracy of the multi-class result vector

    The vector result will be compared to the expected values of this Tester's instance to determine its accuracy based on the function

         (TP + TN)
    -------------------
    (TP + TN + FP + FN) 

    The score is weighted by transforming the given result matrix with the given k-values and finding the accuracy for each class type, then multiplying each score by a weight determined by the frequency of the expected vector's class type (see transform())
    """
    def get_accur(self, result):
        r_matrix, r_weight = self.transform(result)
        acc = (self.get_tp(r_matrix) + self.get_tn(r_matrix)) / np.shape(r_matrix)[1]
        return np.sum(acc * self.weights)
    
    """
    Function to determine the weighted f1-score of the multi-class result vector

    The vector result will be compared to the expected values of this Tester's instance to determine its f1-score based on the function

         (2*TP)
    ---------------
    (2*TP + FP + FN) 

    The score is weighted by transforming the given result matrix with the given k-values and finding the f1-score for each class type, then multiplying each score by a weight determined by the frequency of the expected vector's class type (see transform())
    """
    def get_f1(self, result):
        r_matrix, r_weight = self.transform(result)
        tp = self.get_tp(r_matrix) * 2
        fp = self.get_fp(r_matrix)
        fn = self.get_fn(r_matrix)

        for i in range(self.k):
            if tp[i] == 0 and fp[i] == 0 and fn[i] == 0:
                fp[i] = 1

        score = tp / (tp + fp + fn)
        return np.sum(score * self.weights)
      
    """
    Function to determine the weighted jaccard coefficient of the multi-class result vector

    The vector result will be compared to the expected values of this Tester's instance to determine its jaccard coefficient based on the function

         (TP)
    -------------
    (TP + FP + FN) 

    The score is weighted by transforming the given result matrix with the given k-values and finding the jaccard coefficient for each class type, then multiplying each score by a weight determined by the frequency of the expected vector's class type (see transform())
    
    """  
    def get_jaccard(self, result):
        r_matrix, r_weight = self.transform(result)
        tp = self.get_tp(r_matrix)
        fp = self.get_fp(r_matrix)
        fn = self.get_fn(r_matrix)

        for i in range(self.k):
            if tp[i] == 0 and fp[i] == 0 and fn[i] == 0:
                fp[i] = 1
        
        score = tp / (tp + fp + fn)
        return np.sum(score * self.weights)
    
    """
    Function to determine the accuracy of the result not by the absolute determination of whether or not the result is the exact expected class but by how far the result was from the expected class.

    It is determined by this function:

    ((k-1) - |expected - result|)
    -----------------------------
              (k-1)
    """
    def get_dist_acc(self, result):
        score = ((self.k - 1) - np.absolute(self.expected-result)) / (self.k - 1)
        w = np.ones(np.shape(self.expected)[0]) / (np.shape(self.expected)[0])
        return np.sum(score * w)
    
    """
    Function to determine the accuracy of the result not by the absolute determination of whether or not the result is the exact expected class but by how similar the result was from the expected class (using cosine-similarity)
    """
    def get_sim_acc(self, result):
        dist_r = self.eucl_dist(result * self.step_size)
        return np.dot(self.expected * self.step_size, result * self.step_size)/ (dist_r * self.exp_dist)