import numpy as np
from sklearn.metrics import accuracy_score
from autograd import grad

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

    def feed_forward_batch(self, x):
        a = x
        for (W, b), activation_func in zip(self.layers, self.activation_funcs):
            z = a @ W + b 
            a = activation_func(z)
        return a
    
    def predict(self, x):
        probs = self.feed_forward_batch(x)
        self.predictions = probs
        self._accuracy()
        print("Prediction accuracy: " + str(self.prediction_accuracy))
        return np.argmax(probs, axis = 1)
    
    def _accuracy(self):
        one_hot_predictions = np.zeros(self.predictions.shape)
        for i, prediction in enumerate(self.predictions):
            one_hot_predictions[i, np.argmax(prediction)] = 1
        self.prediction_accuracy = accuracy_score(one_hot_predictions, self.targets)
    
    """ Suggested cost from week 42 exercises

    def cross_entropy(predict, target):
        return np.sum(-target * np.log(predict))

    def cost(input, layers, activation_funcs, target):
        predict = feed_forward_batch(input, layers, activation_funcs)
        return cross_entropy(predict, target)
    """

    def train_network(self, train_input, train_targets, cost, learning_rate=0.001, epochs=100):
        gradient_func = grad(cost, 1)
        for i in range(epochs):
            layers_grad = gradient_func(train_input, self.layers, self.activation_funcs, train_targets)
            i = 0
            for (W, b), (W_g, b_g) in zip(self.layers, layers_grad):
                W -= learning_rate * W_g
                b -= learning_rate * b_g
                self.layers[i] = (W, b)
                i += 1