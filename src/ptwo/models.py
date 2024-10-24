import numpy as np
from sklearn.metrics import accuracy_score
from autograd import grad

class NeuralNetwork:
    """
    Neural Network model

    Args: 
    - 
    
    """
    def __init__(self, network_input, targets, layer_output_sizes, activation_funcs):
        self.network_input = network_input
        self.network_input_size = network_input.size
        self.layer_output_sizes = layer_output_sizes
        self.activation_funcs = activation_funcs
        self.targets = targets

    def create_layers_batch(self):
        self.layers = []
        i_size = self.network_input_size
        for layer_output_size in self.layer_output_sizes:
            W = np.random.randn(i_size, layer_output_size)
            b = np.random.randn(layer_output_size)
            self.layers.append((W, b))
            i_size = layer_output_size

    def feed_forward_batch(self, x):
        a = x
        for (W, b), activation_func in zip(self.layers, self.activation_funcs):
            z = a @ W + b 
            a = activation_func(z)
        return a
    
    def predict_proba(self, x):
        probs = self.feed_forward_batch(x)
        self.predictions = probs
        self._accuracy()
        print("Prediction accuracy: " + str(self.prediction_accuracy))
        return np.argmax(probs, axis = 1)
    
    def predict(self, x):
        probs = self.feed_forward_batch(x)
        self.predictions = probs
        self._accuracy()
        print("Prediction accuracy: " + str(self.prediction_accuracy))
        return np.argmax(probs, axis = 1)
    
    def _accuracy(self):
        one_hot_predictions = np.zeros(self.predictions.shape)
        for i, prediction in enumerate(self.predictions):
            one_hot_predictions[i, np.argmax(prediction)] = 1
        self.prediction_accuracy = accuracy_score(one_hot_predictions, self.targets)
    
    """ Suggested cost from week 42 exercises

    def cross_entropy(predict, target):
        return np.sum(-target * np.log(predict))

    def cost(input, layers, activation_funcs, target):
        predict = feed_forward_batch(input, layers, activation_funcs)
        return cross_entropy(predict, target)
    """

    def train_network(self, train_input, train_targets, cost, learning_rate=0.001, epochs=100):
        gradient_func = grad(cost, 1)
        for i in range(epochs):
            layers_grad = gradient_func(train_input, self.layers, self.activation_funcs, train_targets)
            i = 0
            for (W, b), (W_g, b_g) in zip(self.layers, layers_grad):
                W -= learning_rate * W_g
                b -= learning_rate * b_g
                self.layers[i] = (W, b)
                i += 1

# Retrieved from additionweek42.ipynb
class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.beta_logreg = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def GDfit(self, X, y):
        n_data, num_features = X.shape
        self.beta_logreg = np.zeros(num_features)

        for _ in range(self.num_iterations):
            linear_model = X @ self.beta_logreg
            y_predicted = self.sigmoid(linear_model)

            # Gradient calculation
            gradient = (X.T @ (y_predicted - y))/n_data
            # Update beta_logreg
            self.beta_logreg -= self.learning_rate*gradient

    def predict(self, X):
        linear_model = X @ self.beta_logreg
        y_predicted = self.sigmoid(linear_model)
        return [1 if i >= 0.5 else 0 for i in y_predicted]
    
"""
# Example usage
if __name__ == "__main__":
    # Sample data
    X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    y = np.array([0, 0, 0, 1])  # This is an AND gate
    model = LogisticRegression(learning_rate=0.01, num_iterations=1000)
    model.GDfit(X, y)
    predictions = model.predict(X)
    print("Predictions:", predictions)

"""


class GradientDescent:
    def __init__(self, learning_rate, gradient, momentum = 0, optimizer = None):
        self.learning_rate = learning_rate
        self.gradient = gradient
        self.momentum = momentum
        self.momentum_change = 0.0
        self.optimizer = optimizer
        self.theta = None
        self.n = None
    def _initialize_vars(self, X):
        self.theta = np.random.randn(X.shape[1], 1)
        self.n = X.shape[0]

    def _gd(self, grad, X, y, current_iter):
        if self.optimizer is None:
            update = self.learning_rate * grad + self.momentum * self.momentum_change
            self.momentum_change = update
        else:
            update = self.optimizer.calculate(self.learning_rate, grad, current_iter)

        return update

    def descend(self, X, y, n_iter=500):
        self._initialize_vars(X)
        for i in range(n_iter):
            grad = self.gradient(X, y, self.theta)
            update = self._gd(grad, X, y, i+1)
            self.theta -= update

    def descend_stochastic(self, X, y, n_epochs = 50, batch_size = 5):
        self._initialize_vars(X)
        n_batches = int(self.n / batch_size)
        xy = np.column_stack([X,y]) # for shuffling x and y together
        for i in range(n_epochs):
            if self.optimizer is not None:
                self.optimizer.reset()
            np.random.shuffle(xy)
            for j in range(n_batches):
                random_index = batch_size * np.random.randint(n_batches)
                xi = xy[random_index:random_index+5, :-1]
                yi = xy[random_index:random_index+5, -1:]
                grad = (1/batch_size) * self.gradient(X, y, self.theta)
                update = self._gd(grad, xi, yi, current_iter = j+1)
                self.theta -= update


    