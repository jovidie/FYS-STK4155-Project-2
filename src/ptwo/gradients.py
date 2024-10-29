import autograd.numpy as np

def grad_OLS():
    def grad_fun(X, y, theta):
        n = y.shape[0]
        return (2.0/n) * X.T @ (X @ theta - y)
    return grad_fun

def grad_ridge(lmb):
    def grad_fun(X, y, theta):
        n = y.shape[0]
        return -(2.0/n) * X.T @ (X @ theta - y) + 2*lmb*theta
    return(grad_fun)