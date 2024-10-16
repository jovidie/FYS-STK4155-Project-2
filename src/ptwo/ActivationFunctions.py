import numpy as np

class ActivationFunction:
    def __init__(self, activation_function):
        activation_dict = {
            "sigmoid": self._sigmoid,
            "ReLU": self._ReLU,
            "softmax": self._softmax,
            "softmax_vec": self._softmax_vec
        }

        self.activation_function = activation_dict[activation_function]

    # Activation functions retrieved from exercises week 42
    def _ReLU(z):
        return np.where(z > 0, z, 0)

    def _sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def _softmax(z):
        """Compute softmax values for each set of scores in the rows of the matrix z.
        Used with batched input data."""
        e_z = np.exp(z - np.max(z, axis=0))
        return e_z / np.sum(e_z, axis=1)[:, np.newaxis]

    def _softmax_vec(z):
        """Compute softmax values for each set of scores in the vector z.
        Use this function when you use the activation function on one vector at a time"""
        e_z = np.exp(z - np.max(z))
        return e_z / np.sum(e_z)