import git
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.datasets import load_breast_cancer

from ptwo.models import LogisticRegression
from ptwo.utils import preprocess_cancer_data
from ptwo.plot import plot_heatmap, plot_loss_acc, plot_confusion, set_plt_params
from ptwo.costfuns import binary_cross_entropy
from ptwo.activators import sigmoid

def test_optimal_params_logreg():
    X, y = preprocess_cancer_data()
    N = 5
    lmbda_range = [-5, -2]
    eta_range = [-4, -1]
    # Check where the acc decreases
    lmbdas = np.logspace(lmbda_range[0], lmbda_range[1], N)
    etas = np.logspace(eta_range[0], eta_range[1], N)


    acc_train = np.zeros((N, N))
    acc_test = np.zeros((N, N))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    for i, eta in tqdm(enumerate(etas)):
        for j, lmbda in enumerate(lmbdas):
            clf = LogisticRegression(lmbda)
            clf.fit(X_train_scaled, y_train, eta=eta)

            y_hat = clf.predict(X_train_scaled)
            acc_train[i, j] = accuracy_score(y_train, y_hat)
            y_pred = clf.predict(X_test_scaled)
            acc_test[i, j] = accuracy_score(y_test, y_pred)

            # print(y_hat.shape, y_pred.shape, y_test.shape)

    plot_heatmap(etas, lmbdas, acc_test, figname="test_logreg")


def test_logreg_sgd():
    dataset = load_breast_cancer()
    X = dataset.data
    y = dataset.target
    eta = 0.01
    n_epochs = 100
    lmbda = 0.0001

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logreg = LogisticRegression()
    logreg.fit(X_train_scaled, y_train, batch_size=50, eta=eta, n_epochs=1000)
    y_pred = logreg.predict(X_test_scaled)
    acc = np.mean(y_pred == y_test)
    print(f"Logistic regression accuracy {acc}")


def compare_models(figname=None):
    dataset = load_breast_cancer()
    X = dataset.data
    y = dataset.target
    eta = 0.01
    n_epochs = 200
    epochs = np.arange(n_epochs)
    lmbda = 0.0001

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logreg_gd = LogisticRegression(lmbda=lmbda)
    logreg_gd.fit(X_train_scaled, y_train, eta=eta, n_epochs=n_epochs)
    y_pred_gd = logreg_gd.predict_bias(X_test_scaled)
    acc_gd = accuracy_score(y_test, y_pred_gd)
    print(f"GD accuracy {acc_gd}")

    logreg_loss = logreg_gd.get_loss
    logreg_acc = logreg_gd.get_accuracies

    logreg_sk = LogReg(fit_intercept=False, max_iter=n_epochs)
    logreg_sk.fit(X_train_scaled, y_train)
    y_pred_sk = logreg_sk.predict(X_test_scaled)
    acc_sk = accuracy_score(y_test, y_pred_sk)
    print(f"SKlearn accuracy {acc_sk}")

    set_plt_params()
    plot_loss_acc(epochs, logreg_loss, logreg_acc, figname="logreg_loss_acc")
    set_plt_params(remove_grid=True)
    plot_confusion(y_pred_gd, y_test, title="Logistic regression", figname="logreg_confusion_matrix")
    plot_confusion(y_pred_sk, y_test, title="Sklearn Logistic regression", figname="sklearn_logreg_confusion_matrix")
    # conf1 = confusion_matrix(y_test, y_pred_gd, labels = [0, 1])
    # ConfusionMatrixDisplay(conf1).plot()
    # plt.title(f"Confusion matrix, Logistic Regression")
    # filename = "logreg_confusion"
    # # plt.savefig(f"./latex/figures/{filename}", bbox_inches = "tight")
    # plt.show()


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    # test_optimal_params_logreg()
    compare_models()
    # test_logreg_sgd()

"""Thought around implementation of Logistic regression:

- Needs to add bias to the implementation
"""