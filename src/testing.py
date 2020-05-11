import numpy as np

class Tester():
    def __init__(self, expected, k):
        self.k = k
        self.matrix, self.weights = self.transform(expected)
        self.onesmat = np.ones(np.shape(self.matrix))

    def transform(self, values):
        matrix = np.empty((0, np.shape(values)[0]), dtype='int')
        weights = np.empty(0)

        for i in range(self.k):
            row = (values == i) * 1
            matrix = np.append(matrix, [row], axis = 0)
            weights = np.append(weights, np.sum(row) / np.shape(values)[0])

        return matrix, weights

    def get_tp(self, mat):
        return np.matmul((self.matrix * mat), np.ones(np.shape(mat)[1]))
    
    def get_tn(self, mat):
        return np.matmul(((self.matrix ^ 1) * (mat ^ 1)), np.ones(np.shape(mat)[1]))

    def get_fp(self, mat):
        return np.matmul(((self.matrix ^ 1) * mat), np.ones(np.shape(mat)[1]))

    def get_fn(self, mat):
        return np.matmul((self.matrix * (mat ^ 1)), np.ones(np.shape(mat)[1]))

    def get_accur(self, result):
        r_matrix, r_weight = self.transform(result)
        acc = (self.get_tp(r_matrix) + self.get_tn(r_matrix)) / np.shape(r_matrix)[1]
        return np.sum(acc * self.weights)
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