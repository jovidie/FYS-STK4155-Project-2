import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score

from ptwo.models import NeuralNetwork, LogisticRegression
from ptwo.utils import preprocess_cancer_data
from ptwo.activators import sigmoid
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
    # dataset = load_breast_cancer()
    # X = dataset.data
    # y = dataset.target
    
    X, y = preprocess_cancer_data()
    n_data, n_features = X.shape
    n_outputs = y.shape
    eta = 0.001
    n_epochs = 10000
    lmbda = 0.001

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n----------------------------------------------------------------------------------------")
    print("[        Comparing performance of logistic regression models to SKlearn's model        ] ")
    print("----------------------------------------------------------------------------------------\n")


    logreg_nn = NeuralNetwork(
        network_input_size=n_features, 
        layer_output_sizes=[1],
        activation_funcs=[sigmoid],
        cost_function=binary_cross_entropy,
        # optimizer=RMSProp(),
        lmb=lmbda,
        classification=True
    )
    logreg_nn.train_network(X_train_scaled, y_train)
    y_pred_nn = logreg_nn.predict_proba(X_test_scaled)
    acc_nn = accuracy_score(y_test, y_pred_nn)
    print(f"NN accuracy {acc_nn}")

    logreg_gd = LogisticRegression(lmbda=lmbda)
    logreg_gd.fit(X_train_scaled, y_train, eta=0.001, n_epochs=n_epochs)
    y_pred_gd = logreg_gd.predict(X_test_scaled)
    acc_gd = accuracy_score(y_test, y_pred_gd)
    print(f"GD accuracy {acc_gd}")

    logreg_sgd = LogisticRegression(lmbda=lmbda)
    logreg_sgd.fit(X_train_scaled, y_train, batch_size=50, optimizer=ADAM(), eta=0.001, n_epochs=n_epochs)
    y_pred_sgd = logreg_sgd.predict(X_test_scaled)
    acc_sgd = accuracy_score(y_test, y_pred_sgd)
    print(f"SGD accuracy {acc_sgd}")

    logreg_sk = LogReg(fit_intercept=False, max_iter=n_epochs)
    logreg_sk.fit(X_train_scaled, y_train)
    y_pred_sk = logreg_sk.predict(X_test_scaled)
    acc_sk = accuracy_score(y_test, y_pred_sk)
    print(f"SKlearn accuracy {acc_sk}")

    

# part f is discussion only

def main():
    pass



if __name__ == '__main__':
    # np.random.seed(2024)
    part_e()