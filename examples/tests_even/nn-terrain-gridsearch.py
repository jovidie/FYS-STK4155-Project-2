import autograd.numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from imageio.v2 import imread

from ptwo.activators import ReLU, sigmoid
from ptwo.models import NeuralNetwork
from ptwo.optimizers import Momentum, ADAM, AdaGrad, RMSProp
from ptwo.costfuns import mse
from ptwo.utils import lambda_lr_heatmap

terrain_full = imread('data/SRTM_data_Norway_2.tif')
# subset to a manageable size
terrain1 = terrain_full[1050:1250, 500:700]

x_1d = np.arange(terrain1.shape[1])
y_1d = np.arange(terrain1.shape[0])
# create grid
x_2d, y_2d = np.meshgrid(x_1d,y_1d)

# flatten the data and features
X = np.column_stack((x_2d.flatten(), y_2d.flatten()))

y = np.asarray(terrain1.flatten())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
y_test = y_test[:,np.newaxis]
y_train = y_train[:,np.newaxis]

scalerX = StandardScaler(with_std = True)
X_train_scaled = scalerX.fit_transform(X_train)
X_test_scaled = scalerX.transform(X_test)

scalery = StandardScaler(with_std = True)
y_train_scaled = scalery.fit_transform(y_train)
y_test_scaled = scalery.transform(y_test)

## Test neural network
np.random.seed(4927384)
input_size = X_train.shape[1]
layer_output_sizes = [20, 10, 1]
activation_funs = [sigmoid, ReLU, lambda x: x]
#learning_rate=0.1

# nn = NeuralNetwork(input_size, layer_output_sizes, activation_funs, mse)
# print(nn.get_cost(X_test_scaled, y_test_scaled))
# nn.train_network(X_train_scaled, y_train_scaled, learning_rate, verbose=True, epochs=301)
# print(nn.get_cost(X_test_scaled, y_test_scaled))

# np.random.seed(4927384)
# learning_rate=0.1
# nn = NeuralNetwork(input_size, layer_output_sizes, activation_funs, mse, optimizer=ADAM())
# print(nn.get_cost(X_test_scaled, y_test_scaled))
# nn.train_network(X_train_scaled, y_train_scaled, learning_rate, verbose=True, epochs=301)
# print(nn.get_cost(X_test_scaled, y_test_scaled))

learning_rates = np.logspace(-3, 0, 10)
lmbs = np.logspace(-6, 0, 10)
np.save("examples/tests_even/data/learning_rates.npy", learning_rates)
np.save("examples/tests_even/data/lmbs.npy", lmbs)


mses = np.zeros( (len(lmbs), len(learning_rates)) )
for i, learning_rate in enumerate(learning_rates):
    for j, lmb in enumerate(lmbs):
        np.random.seed(4927384)
        nn = NeuralNetwork(input_size, layer_output_sizes, activation_funs, mse, optimizer=ADAM(), lmb=lmb)
        nn.train_network(X_train_scaled, y_train_scaled, learning_rate, epochs=301)
        mses[j, i] = nn.get_cost(X_test_scaled, y_test_scaled)

np.save("examples/tests_even/data/mses-terrain-gd-adam.npy", mses)


mses_sgd = np.zeros( (len(lmbs), len(learning_rates)) )
for i, learning_rate in enumerate(learning_rates):
    for j, lmb in enumerate(lmbs):
        np.random.seed(4927384)
        nn = NeuralNetwork(input_size, layer_output_sizes, activation_funs, mse, optimizer=ADAM(), lmb=lmb)
        nn.train_network(X_train_scaled, y_train_scaled, learning_rate, epochs=50, batch_size=32)
        mses_sgd[j, i] = nn.get_cost(X_test_scaled, y_test_scaled)

np.save("examples/tests_even/data/mses-terrain-sgd-adam.npy", mses_sgd)


lambda_lr_heatmap(mses, lmbs, learning_rates)

lambda_lr_heatmap(mses_sgd, lmbs, learning_rates)