import numpy as np

class Tester():
    def __init__(self, expected, k):
        self.k = k
        self.matrix, self.weights = self.transform(expected)

    def transform(self, values):
        matrix = np.empty((0, np.shape(values)[0]))
        weights = np.empty(0)

        for i in range(self.k):
            row = (values == i) * 1
            matrix = np.append(matrix, [row], axis = 0)
            weights = np.append(weights, np.sum(row) / self.k)

        return matrix, weights

    def get_tp(self, mat):
        return np.matmul((self.matrix * mat), np.ones(np.shape(mat)[0]))
    
    def get_tn(self, mat):
        return np.matmul((self.matrix + mat) ^ 1, np.ones(np.shape(mat)[0]))

    def get_accur(self, result):
        r_matrix, r_weight = self.transform(result)

        
        return
    def get_f1(self, result):
        return
    def get_jaccob(self, result):
        return