import numpy as np
from sklearn.neural_network import MLPRegressor as mlpc

class NeuralNetwork():
    def __init__(self, solver = "adam", activation = "relu", hidden_layer_sizes = None):
        if hidden_layer_sizes == None:
            hidden_layer_sizes = (100,)
        self.regressor = mlpc(solver = solver, activation = activation, hidden_layer_sizes = hidden_layer_sizes)
    
    def setUp(self, matrix, values):
        self.regressor.fit(matrix, values)
    
    def predict(self, matrix):
        return self.regressor.predict(matrix)

        
