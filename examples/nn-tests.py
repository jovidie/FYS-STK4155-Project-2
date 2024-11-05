from ptwo.models import NeuralNetwork
from ptwo.activators import sigmoid, ReLU, softmax
from ptwo.optimizers import ADAM, AdaGrad, RMSProp, Momentum
from ptwo.costfuns import mse, cross_entropy

import autograd.numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split


# generate data

np.random.seed(8923)

# generate data
def data_function(x):
    return 4*x**3 + x**2 - 17*x + 48

n = 100
x = np.linspace(-2, 2, n)
f_x = data_function(x)
y = f_x + np.random.normal(0, 1, n)
y = y.reshape(-1,1)

# create design matrix of polynomials
# for now 3rd order reflecting data function
X = PolynomialFeatures(3).fit_transform(x.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

scalerx = StandardScaler()
X_train_scaled = scalerx.fit_transform(X_train)
X_test_scaled = scalerx.transform(X_test)

scalery = StandardScaler()
y_train_scaled = scalerx.fit_transform(y_train)
y_test_scaled = scalerx.transform(y_test)

input_size = X_train.shape[1]
layer_output_sizes = [10, 1]
activation_funs = [ReLU, lambda x: x]

print(" ------------------")
print("| Gradient descent |")
print(" ------------------")

nn = NeuralNetwork(input_size, layer_output_sizes, activation_funs, mse)

print("No optimizer")
print("Train MSE before training", nn.get_cost(X_train, y_train))

nn.train_network(X_train, y_train, learning_rate=0.001, epochs=1000)

print("Train MSE after training", nn.get_cost(X_train, y_train))
print("Test MSE after training", nn.get_cost(X_test, y_test))

print("-----------------------------")
print("With momentum:")
gamma = 0.3
nn = NeuralNetwork(input_size, layer_output_sizes, activation_funs, mse, optimizer=Momentum(gamma))

print("Train MSE before training", nn.get_cost(X_train, y_train))

print("MSE before training", mse(X_train, y_train))

nn.train_network(X_train, y_train, learning_rate=0.001, epochs=1000)

print("Train MSE after training", nn.get_cost(X_train, y_train))
print("Test MSE after training", nn.get_cost(X_test, y_test))


print("-----------------------------")
print("With ADAM optimizer:")

nn = NeuralNetwork(input_size, layer_output_sizes, activation_funs, mse, optimizer=ADAM())

print("MSE before training", mse(X_train, y_train))

nn.train_network(X_train, y_train, learning_rate=0.5, epochs=1000)
print("Train MSE after training", nn.get_cost(X_train, y_train))
print("Test MSE after training", nn.get_cost(X_test, y_test))
print("-----------------------------")

print("With AdaGrad optimizer")
nn = NeuralNetwork(input_size, layer_output_sizes, activation_funs, mse, optimizer=AdaGrad())

print("MSE before training", mse(X_train, y_train))

nn.train_network(X_train, y_train, learning_rate=0.5, epochs=1000)
print("Train MSE after training", nn.get_cost(X_train, y_train))
print("Test MSE after training", nn.get_cost(X_test, y_test))

print("-----------------------------")

print("With RMSProp optimizer")
rho = 0.99
nn = NeuralNetwork(input_size, layer_output_sizes, activation_funs, mse, optimizer=RMSProp(rho))

print("MSE before training", mse(X_train, y_train))

nn.train_network(X_train, y_train, learning_rate=0.01, epochs=1000)
print("Train MSE after training", nn.get_cost(X_train, y_train))
print("Test MSE after training", nn.get_cost(X_test, y_test))

print(" -----------------------------")
print("| Stochastic gradient descent |")
print(" -----------------------------")

print("No optimizer")
nn = NeuralNetwork(input_size, layer_output_sizes, activation_funs, mse)

print("MSE before training", mse(X_train, y_train))
nn.train_network(X_train, y_train, learning_rate=0.001, epochs=100, batch_size=5)
print(nn.optimizer)
print("Train MSE after training", nn.get_cost(X_train, y_train))
print("Test MSE after training", nn.get_cost(X_test, y_test))

print("-----------------------------")
print("With momentum:")
gamma = 0.3
nn = NeuralNetwork(input_size, layer_output_sizes, activation_funs, mse, optimizer=Momentum(gamma))

print("MSE before training", mse(X_train, y_train))

nn.train_network(X_train, y_train, learning_rate=0.001, epochs=100, batch_size=5)
print("Train MSE after training", nn.get_cost(X_train, y_train))
print("Test MSE after training", nn.get_cost(X_test, y_test))

print("-----------------------------")
print("With ADAM:")
gamma = 0.3
nn = NeuralNetwork(input_size, layer_output_sizes, activation_funs, mse, optimizer=ADAM())

print("MSE before training", mse(X_train, y_train))

nn.train_network(X_train, y_train, learning_rate=0.01, epochs=100, batch_size=5)
print("Train MSE after training", nn.get_cost(X_train, y_train))
print("Test MSE after training", nn.get_cost(X_test, y_test))



print("-----------------------------")
print("With ADAM and L2 regularization:")
gamma = 0.3
nn = NeuralNetwork(input_size, layer_output_sizes, activation_funs, mse, optimizer=ADAM(), lmb=0.1)

print("MSE before training", mse(X_train, y_train))

nn.train_network(X_train, y_train, learning_rate=0.01, epochs=100, batch_size=5)
print("Train MSE after training", nn.get_cost(X_train, y_train))
print("Test MSE after training", nn.get_cost(X_test, y_test))

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

nn.train_network(input_train, target_train, learning_rate=0.01, epochs=300, batch_size=5)

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

nn.train_network(input_train, target_train, learning_rate=0.01, epochs=300)


print("Accuracy after training")
print(nn.accuracy(input_train, target_train))
print("Test accuracy")
print(nn.accuracy(input_test, target_test))