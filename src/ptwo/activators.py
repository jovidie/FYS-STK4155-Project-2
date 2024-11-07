import autograd.numpy as np

# Activation functions retrieved from exercises week 42
def ReLU(z):
    return np.where(z > 0, z, 0)

def relu6(x):
    return np.clip(x, 0, 6)

def leaky_ReLU(z):
    # week 42 notes
    delta = 10e-4
    return np.where(z > np.zeros(z.shape), z, delta * z)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(z):
    """Compute softmax values for each set of scores in the rows of the matrix z.
    Used with batched input data."""
    e_z = np.exp(z - np.max(z, axis=0))
    return e_z / np.sum(e_z, axis=1)[:, np.newaxis]

def softmax_vec(z):
    """Compute softmax values for each set of scores in the vector z.
    Use this function when you use the activation function on one vector at a time"""
    e_z = np.exp(z - np.max(z))
    return e_z / np.sum(e_z)