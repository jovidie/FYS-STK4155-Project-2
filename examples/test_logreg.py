import git
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from ptwo.models import LogisticRegression
from ptwo.utils import preprocess_cancer_data
from ptwo.plot import plot_heatmap

def test_optimal_params_logreg():
    df, X, y = preprocess_cancer_data()
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

    plot_heatmap(etas, lmbdas, acc_test, figname="test_logreg.pdf")


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    test_optimal_params_logreg()