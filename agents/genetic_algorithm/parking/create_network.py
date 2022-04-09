import numpy as np
from genetic_algorithm.parking.base_nn import neural_network


class activation_func():
    def ReLU_function(x):
        return np.maximum(x, 0)

    def sigmoid_function(x):
        return 1 / (1 + np.exp(-x))


class create_network(neural_network):

    def __init__(self, input_size, hidden_layer, output_size):
        self.input_size = input_size
        self.hidden_layer = hidden_layer
        self.output_size = output_size
        self.weights1 = np.random.randn(self.input_size, self.hidden_layer)
        self.biases1 = np.random.randn(self.hidden_layer)
        self.weights2 = np.random.randn(self.hidden_layer, self.output_size)
        self.biases2 = np.random.rand(self.output_size)

    def forward(self, output: np.array) -> np.array:
        output = output @ self.weights1 + self.biases1
        output = activation_func.ReLU_function(output)
        output = output @ self.weights2 + self.biases2
        return activation_func.sigmoid_function(output)

    def retrieve_weight_and_bias(self):
        return np.concatenate((self.weights1.flatten(), self.biases1, self.weights2.flatten(), self.biases2), axis=0)

    def weight_bias_reform(self, weights_biases: np.array) -> None:
        weight1, bias1, weight2, bias2 = np.split(weights_biases,
                                                  [self.weights1.size, self.weights1.size + self.biases1.size,
                                                   self.weights1.size + self.biases1.size + self.weights2.size])
        self.weights1 = np.resize(
            weight1, (self.input_size, self.hidden_layer))
        self.biases1 = bias1
        self.weights2 = np.resize(
            weight2, (self.hidden_layer, self.output_size))
        self.biases2 = bias2
