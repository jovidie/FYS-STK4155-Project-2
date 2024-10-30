import autograd.numpy as np

def mse(predict, target):
    return np.mean((predict - target) ** 2)

