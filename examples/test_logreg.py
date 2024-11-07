import git
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from ptwo.models import LogisticRegression, NeuralNetwork
from ptwo.utils import preprocess_cancer_data
from ptwo.plot import plot_heatmap
from ptwo.costfuns import binary_cross_entropy
from ptwo.activators import sigmoid

def test_optimal_params_logreg():
    X, y = preprocess_cancer_data()
    N = 100
    lmbda_range = [-3, -2]
    eta_range = [-6, -2]
    # Check where the mse decreases
    lmbdas = np.logspace(lmbda_range[0], lmbda_range[1], N)
    etas = np.logspace(eta_range[0], eta_range[1], N)

    acc_train = np.zeros((N, N))
    acc_test = np.zeros((N, N))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    for i, eta in tqdm(enumerate(etas)):
        for j, lmbda in enumerate(lmbdas):
            clf = LogisticRegression()
            clf.fit(X_train, y_train, eta=eta)

            y_hat = clf.predict(X_train)
            acc_train[i, j] = accuracy_score(y_train, y_hat)
            y_pred = clf.predict(X_test)
            acc_test[i, j] = accuracy_score(y_test, y_pred)

            # print(y_hat.shape, y_pred.shape, y_test.shape)

    plot_heatmap(etas, lmbdas, acc_test, figname="test_logreg")


def compare_models():
    X, y = preprocess_cancer_data()
    n_data, n_features = X.shape
    n_outputs = y.shape
    eta = 0.001

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logreg_nn = NeuralNetwork(
        network_input_size=n_features, 
        layer_output_sizes=[1],
        activation_funcs=[sigmoid],
        cost_function = binary_cross_entropy,
        classification = True
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


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    # test_optimal_params_logreg()
    compare_models()