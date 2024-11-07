import autograd.numpy as np

def mse(predict, target):
    return np.mean((predict - target) ** 2)

def cross_entropy(predict, target):
    return np.sum(-target * np.log(predict + 1e-15))

def binary_cross_entropy(predict, target):
    """
    Loss function used in binary classification when target variable has two possible 
    outcomes: 1 or 0, 
    Args: 
    - predict is the prediction we have from input
    - target are the targets we know to match input
"""
    return - np.mean(target * np.log(predict + 1e-15) + (1 - target) * np.log(1 - predict + 1e-15))

