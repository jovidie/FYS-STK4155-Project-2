from ptwo.models import NeuralNetwork
from ptwo.activators import sigmoid, ReLU, softmax
from ptwo.optimizers import ADAM, AdaGrad, RMSProp, Momentum
from ptwo.costfuns import mse, cross_entropy

import autograd.numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split


# generate data



print(" -------------------------------------")
print("| Classification on the iris data set |")
print(" -------------------------------------")

from sklearn import datasets
from sklearn.metrics import accuracy_score

# load data
iris = datasets.load_iris()
inputs = iris.data

targets = np.zeros((len(iris.data), 3))
for i, t in enumerate(iris.target):
    targets[i, t] = 1

network_input_size = inputs.shape[1]
layer_output_sizes = [8, 3]
activation_funcs = [sigmoid, softmax]

input_train, input_test, target_train, target_test = train_test_split(inputs, targets, test_size=0.8)

# test neural networks
print("Regular GD")
nn = NeuralNetwork(network_input_size, layer_output_sizes, activation_funcs, cross_entropy)

print("Accuracy before training")
print(nn.accuracy(input_train, target_train))

nn.train_network(input_train, target_train, learning_rate=0.001, epochs=1000)

print("Accuracy after training")
print(nn.accuracy(input_train, target_train))
print("Test accuracy")
print(nn.accuracy(input_test, target_test))

print("-------------------------------")
print("SGD")
nn = NeuralNetwork(network_input_size, layer_output_sizes, activation_funcs, cross_entropy)
print("Accuracy before training")
print(nn.accuracy(input_train, target_train))

nn.train_network(input_train, target_train, learning_rate=0.01, epochs=1000, batch_size=5)

print("Accuracy after training")
print(nn.accuracy(input_train, target_train))
print("Test accuracy")
print(nn.accuracy(input_test, target_test))

print("-------------------------------")
print("GD with regularization")
nn = NeuralNetwork(network_input_size, layer_output_sizes, activation_funcs, cross_entropy, lmb=0.1)
print("Accuracy before training")
print(nn.accuracy(input_train, target_train))

nn.train_network(input_train, target_train, learning_rate=0.001, epochs=1000)

print("Accuracy after training")
print(nn.accuracy(input_train, target_train))
print("Test accuracy")
print(nn.accuracy(input_test, target_test))

print("-------------------------------")
print("SGD with regularization and ADAM")
nn = NeuralNetwork(network_input_size, layer_output_sizes, activation_funcs, cross_entropy, optimizer=ADAM(), lmb=0.1)
print("Accuracy before training")
print(nn.accuracy(input_train, target_train))

nn.train_network(input_train, target_train, learning_rate=0.01, epochs=1000)

print("Accuracy after training")
print(nn.accuracy(input_train, target_train))
print("Test accuracy")
print(nn.accuracy(input_test, target_test))