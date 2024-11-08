import git
import autograd.numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ptwo.models import GradientDescent
from ptwo.gradients import grad_ridge
from ptwo.costfuns import mse
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('future.no_silent_downcasting', True)


def calculate_polynomial(x, *args):
    """
    Calculate an arbitrary polynomial expression
    Args:
    - x: input value
    - *args: values for each polynomial degree:
    Returns: 
        args[0] + args[1]*x**1 + args[2]*x**2 + ... + args[n]*x**n
    Examples:
        x = np.random.randn(10)
        print(calculate_polynomial(x, 10, 2, -5))
        # or with a list:
        arglist = [10, 2, -5]
        print(calculate_polynomial(x, *arglist))
    """

    y = 0
    for i in range(len(args)):
        y += args[i] * x**i
    return y

def franke_function(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

# utilities for doing and plotting grid search with gradient descent

def GD_lambda_mse(
    X_train, X_test, y_train, y_test, learning_rate, lmbs, n_iter, batch_size=None, optimizer=None, gradient_fun=grad_ridge):
    mses=np.zeros(len(lmbs))
    for i in range(len(lmbs)):
        gd = GradientDescent(learning_rate, gradient_fun(lmbs[i]), optimizer=optimizer)
        gd.descend(X_train, y_train, n_iter, batch_size)
        y_pred = X_test @ gd.theta
        mses[i] = mse(y_test, y_pred)
    return mses

def eta_lambda_grid(
    X_train, X_test, y_train, y_test, learning_rates, lmbs, n_iter, batch_size=None, optimizer=None, gradient_fun=grad_ridge
    ):
    mses = np.zeros( (len(lmbs), len(learning_rates)) )
    for i in range(len(learning_rates)):
        mse_eta = GD_lambda_mse(
            X_train, X_test, y_train, y_test, learning_rate=learning_rates[i], lmbs=lmbs, n_iter=n_iter, batch_size=batch_size, optimizer=optimizer, gradient_fun=gradient_fun
            )
        mses[:,i] = mse_eta
    return mses

def lambda_lr_heatmap(mses, lmbs, learning_rates, lmb_label_res=3, lr_label_res=3, filename=None):
    lmb_lab = ["{0:.2e}".format(x) for x in lmbs]
    lr_lab = ["{0:.2e}".format(x) for x in learning_rates]

    sns.heatmap(mses, annot=True)
    plt.xticks(np.arange(len(learning_rates))[1::lmb_label_res] + 0.5, lr_lab[1::lmb_label_res])
    plt.yticks(np.arange(len(lmbs))[1::lr_label_res] + 0.5, lmb_lab[1::lr_label_res])
    plt.xlabel(r"Learning rate $\eta$")
    plt.ylabel(r"$\lambda$")
    if filename is not None:
        plt.savefig(filename)
    plt.show()


def preprocess_cancer_data(save=False):
    ROOT = git.Repo(".", search_parent_directories=True).working_dir
    filename = "wisconsin_breast_cancer_data.csv"
    df = pd.read_csv(f"{ROOT}/data/{filename}")
    print(df)
    # Clean up
    not_include = ["id", "diagnosis", "Unnamed: 32"]
    input_features = df[df.columns.difference(not_include)]
    print(input_features)
    target_feature = df.diagnosis
    print(target_feature)
    # Transform to binary
    pd.set_option('future.no_silent_downcasting', True)
    target_feature.replace({"M":1, "B":0}, inplace=True)
    print(target_feature)

    if save:
        dataset = pd.concat([input_features, target_feature], axis=1).reindex(input_features.index)
        save_as = "wisconsin_breast_cancer_data_binary_target.csv"
        dataset.to_csv(f"{ROOT}/data/{save_as}.csv")

    else:
        # X = np.array(input_features, dtype=float)
        # y = np.array(target_feature, dtype=float)
        X = input_features.to_numpy()
        print(X[:, :2])
        y = target_feature.to_numpy()
        return X, y
    

if __name__ == '__main__':
    pass
    # preprocess_cancer_data(save=True)