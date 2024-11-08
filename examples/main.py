import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score

from ptwo.models import NeuralNetwork, LogisticRegression
from ptwo.utils import preprocess_cancer_data
from ptwo.activators import relu6, sigmoid
from ptwo.costfuns import binary_cross_entropy
from ptwo.optimizers import ADAM, AdaGrad, RMSProp

def part_a():
    # GD with fixed learning rate and analytical gradient
    # Compare GD with momentum and analytical gradient
    # SGD with tunable learning rate, and analytical gradient
    # Study result as function of batch size, number of epochs
    # Add Adagrad RMSprop and Adam and tune learning rate
    # Replace analytical gradient with autograd and compare
    # Compare with Scikit-learn
    pass

def part_b():
    # Neural network and choice of cost function
    # Regression analysis of terrain data, with sigmoid in hidden layers
    # Choose init of weights and biases, as well as output activation
    # Compare results with linear regression from project 1
    # Compare with Scikit-learn
    # Study optimal MSE and R2 as function of learning rate and regularization param
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
    part_e()