import numpy as np
from sklearn.metrics import accuracy_score

class NeuralNetwork:
    def __init__(self, network_input, targets, layer_output_sizes, activation_funcs):
        self.network_input = network_input
        self.network_input_size = network_input.size
        self.layer_output_sizes = layer_output_sizes
        self.activation_funcs = activation_funcs
        self.targets = targets

    def create_layers_batch(self):
        self.layers = []
        i_size = self.network_input_size
        for layer_output_size in self.layer_output_sizes:
            W = np.random.randn(i_size, layer_output_size)
            b = np.random.randn(layer_output_size)
            self.layers.append((W, b))
            i_size = layer_output_size

    def feed_forward_batch(self):
        a = self.network_input
        for (W, b), activation_func in zip(self.layers, self.activation_funcs):
            z = a @ W + b 
            a = activation_func(z)
        return a
    
    def accuracy(self):
        one_hot_predictions = np.zeros(self.predictions.shape)
        for i, prediction in enumerate(self.predictions):
            one_hot_predictions[i, np.argmax(prediction)] = 1
        return accuracy_score(one_hot_predictions, self.targets)
        