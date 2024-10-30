from ptwo.models import NeuralNetwork
from ptwo.activators import sigmoid, ReLU
from ptwo.optimizers import ADAM

import autograd.numpy as np
from autograd import grad
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split

def ReLU(z):
    return np.where(z > 0, z, 0)


def mse(predict, target):
    return np.mean((predict - target) ** 2)

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

nn = NeuralNetwork(X_train.shape[1], layer_output_sizes, activation_funs, mse)

out = nn.feed_forward_batch(X_train)

print("MSE before training", mse(out, y_train))



nn.train_network(X_train, y_train, learning_rate=0.001, epochs=1000)
out = nn.feed_forward_batch(X_train)

print("MSE after training", mse(out, y_train))

print("With ADAM optimizer:")
# for now until I become smarter
def create_layers_batch(network_input_size, layer_output_sizes):
    layers = []

    i_size = network_input_size
    for layer_output_size in layer_output_sizes:
        W = np.random.randn(i_size, layer_output_size)
        b = np.random.randn(layer_output_size)
        layers.append((W, b))

        i_size = layer_output_size
    return layers

layers = create_layers_batch(X_train.shape[1], layer_output_sizes)

nn = NeuralNetwork(X_train.shape[1], layer_output_sizes, activation_funs, mse, optimizer=ADAM(layers))

out = nn.feed_forward_batch(X_train)

print("MSE before training", mse(out, y_train))

nn.train_network(X_train, y_train, learning_rate=0.5, epochs=1000)
out = nn.feed_forward_batch(X_train)

print("MSE after training", mse(out, y_train))
