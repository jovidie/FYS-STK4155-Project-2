import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score

from imageio.v2 import imread

from ptwo.models import NeuralNetwork, LogisticRegression, GradientDescent
from ptwo.utils import preprocess_cancer_data
from ptwo.activators import relu6, sigmoid, ReLU
from ptwo.costfuns import binary_cross_entropy, mse
from ptwo.optimizers import ADAM, AdaGrad, RMSProp
from ptwo.gradients import grad_ridge


def part_a():
    terrain_full = imread('data/SRTM_data_Norway_2.tif')
    # subset to a manageable size
    terrain1 = terrain_full[1050:1250, 500:700]

    x_1d = np.arange(terrain1.shape[1])
    y_1d = np.arange(terrain1.shape[0])
    # create grid
    x_2d, y_2d = np.meshgrid(x_1d,y_1d)

    # flatten the data and features
    xy = np.column_stack((x_2d.flatten(), y_2d.flatten()))
    max_poly = 10
    X_feat = PolynomialFeatures(max_poly).fit_transform(xy)
    X = X_feat[:, 1:]

    y = terrain1.flatten()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    y_test = y_test[:,np.newaxis]
    y_train = y_train[:,np.newaxis]

    scalerX = StandardScaler(with_std = True)
    X_train_scaled = scalerX.fit_transform(X_train)
    X_test_scaled = scalerX.transform(X_test)

    scalery = StandardScaler(with_std = True)
    y_train_scaled = scalery.fit_transform(y_train)
    y_test_scaled = scalery.transform(y_test)

    # results from grid search applied to the terrain data

    np.random.seed(2309148230)
    learning_rate=1.78e-3
    n_iter = 1000
    lmb=1.78e-6
    grad = grad_ridge(lmb)
    #grad = grad_OLS()
    gd = GradientDescent(learning_rate, grad,  optimizer = ADAM())
    gd.descend(X_train_scaled, y_train_scaled, n_iter, batch_size = 32)
    pred = X_test_scaled@gd.theta
    print("Final MSE of SGD on terrain data:", mse(pred, y_test_scaled))


def part_b():
    # Neural network and choice of cost function
    # Regression analysis of terrain data, with sigmoid in hidden layers
    # Choose init of weights and biases, as well as output activation
    # Compare results with linear regression from project 1
    # Compare with Scikit-learn
    # Study optimal MSE and R2 as function of learning rate and regularization param
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


    ## Final test with optimal values
    np.random.seed(65345)
    input_size = X_train.shape[1]
    layer_output_sizes = [20, 10, 1]
    activation_funs = [sigmoid, ReLU, lambda x: x]
    learning_rate=1.67e-02
    lmb = 1e-9

    nn = NeuralNetwork(input_size, layer_output_sizes, activation_funs, mse, optimizer=ADAM())
    nn.train_network(X_train_scaled, y_train_scaled, learning_rate, epochs=50, batch_size=32)
    print("Final MSE of NN on terrain data:", nn.get_cost(X_test_scaled, y_test_scaled))

    pass

def part_c():
    # Test different activation functions for hidden layers
    # Compare results from Sigmoid, ReLU, and Leaky ReLU
    # Study the effect of methods for init weights and biases 
    pass

def part_d():
    # Change activation for output and cost function to do classification
    # Study performance/accuracy as function of learning rate, regularization param,
    # activation functions, number of hidden layers and nodes
    # Compare with Scikit-learn
    pass

def part_e():
    # Classification with logistic regression
    # Study the results of logreg as function of learning rate and regularizarion param
    # Compare with FFNN and Scikit-learn
    dataset = load_breast_cancer()
    X = dataset.data
    y = dataset.target
    eta = 0.01
    n_epochs = 20
    lmbda = 0.0001

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n----------------------------------------------------------------------------------------")
    print("[        Comparing performance of logistic regression model to SKlearn's model         ] ")
    print("----------------------------------------------------------------------------------------\n")

    logreg_gd = LogisticRegression(lmbda=lmbda)
    logreg_gd.fit(X_train_scaled, y_train, eta=eta, n_epochs=n_epochs)
    y_pred_gd = logreg_gd.predict_bias(X_test_scaled)
    acc_gd = accuracy_score(y_test, y_pred_gd)
    print(f"GD accuracy {acc_gd}")

    # logreg_loss = logreg_gd.get_loss
    # logreg_acc = logreg_gd.get_accuracies

    logreg_sk = LogReg(fit_intercept=False, max_iter=n_epochs)
    logreg_sk.fit(X_train_scaled, y_train)
    y_pred_sk = logreg_sk.predict(X_test_scaled)
    acc_sk = accuracy_score(y_test, y_pred_sk)
    print(f"SKlearn accuracy {acc_sk}")

# part f is discussion only

def main():
    pass



if __name__ == '__main__':
    np.random.seed(2024)
    part_a()
    part_b()
    part_e()