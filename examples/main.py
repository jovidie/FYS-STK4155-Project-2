import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from ptwo.models import NeuralNetwork, LogisticRegression
from ptwo.utils import preprocess_cancer_data
from ptwo.activators import sigmoid
from ptwo.costfuns import binary_cross_entropy
#from ptwo.optimizers import 

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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logreg_nn = NeuralNetwork(
        network_input_size=n_features, 
        layer_output_sizes=[1],
        activation_funcs=[sigmoid],
        cost_function=binary_cross_entropy,
        classification=True
    )
    logreg_nn.train_network(X_train_scaled, y_train)
    y_prob = logreg_nn.predict(X_test_scaled)
    y_pred = (y_prob>0.5).astype('int')
    # print(y_pred)
    # print(y_test)
    acc = np.mean(y_pred == y_test)
    print(f"NN logreg accuracy {acc}")

    logreg = LogisticRegression()
    logreg.fit(X_train_scaled, y_train, eta=0.001, n_epochs=1000)
    y_pred = logreg.predict(X_test_scaled)
    # print(y_pred)
    # print(y_test)
    acc = np.mean(y_pred == y_test)
    print(f"Logistic regression accuracy {acc}")
    

# part f is discussion only

def main():
    pass

def identity(X):
    return X

class Scheduler:
    """
    Abstract class for Schedulers
    """

    def __init__(self, eta):
        self.eta = eta

    # should be overwritten
    def update_change(self, gradient):
        raise NotImplementedError

    # overwritten if needed
    def reset(self):
        pass


class Constant(Scheduler):
    def __init__(self, eta):
        super().__init__(eta)

    def update_change(self, gradient):
        return self.eta * gradient
    
    def reset(self):
        pass


if __name__ == '__main__':
    part_e()