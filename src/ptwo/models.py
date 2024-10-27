import numpy as np
from sklearn.metrics import accuracy_score
from autograd import grad

class NeuralNetwork:
    """
    Neural Network model

    Args: 
    - network_input: the design matrix/variables we wish to predict from
    - targets: the golden truth
    - layer_output_sizes = size of layers, number of layers is determined by len(layer_output sizes)
    - activation_funcs is the \sigma() function 
    
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
    
    # TODO - fix proba
    def predict_proba(self, x, targets):
        probs = self.feed_forward_batch(x)
        self.predictions = probs
        return np.argmax(probs, axis = 1)
    
    def predict(self, x):
        probs = self.feed_forward_batch(x)
        self.predictions = probs
        return np.argmax(probs, axis = 1)
    
    def accuracy(self, x, targets):
        predictions = self.feed_forward_batch(x)
        one_hot_predictions = np.zeros(predictions.shape)
        for i, prediction in enumerate(predictions):
            one_hot_predictions[i, np.argmax(prediction)] = 1
        self.prediction_accuracy = accuracy_score(one_hot_predictions, targets)
    
    # Suggested cost from week 42 exercises
    def _cross_entropy(self, predict, target):
        return np.sum(-target * np.log(predict))

    def _cost(self):
        self.train_prediction = self.feed_forward_batch(self.train_input)
        return self.cross_entropy(self.train_predict, self.train_target)

    def train_network(self, train_input, train_targets, cost, learning_rate=0.001, epochs=100):
        self.train_input = train_input
        self.train_targets = train_targets
        gradient_func = grad(cost, 1)
        layers_grad = gradient_func(train_input, self.layers, self.activation_funcs, train_targets)  # Don't change this
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
    

# Not necessary with both, condence classes into one
class LogReg:
    def __init__(self, bias=-1):
        self.bias = bias
        self.weights = None
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.epochs_trained = None
    
    @property
    def epochs(self):
        return self.epochs_trained

    @property
    def losses(self):
        return self.train_losses, self.val_losses
    
    @property
    def accuracies(self):
        return self.train_accuracies, self.val_accuracies

    @epochs.setter
    def epochs(self, epoch):
        self.epochs_trained = epoch

    def add_bias(self, X):
        N = X.shape[0]
        biases = np.ones((N, 1)) * self.bias
        return np.concatenate((biases, X), axis  = 1) 

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def accuracy(self, y_pred, y_true, threshold=0.5):
        y_cat = (y_pred > threshold).astype('int')
        return np.mean(y_cat == y_true)
    
    def loss(self, y_pred, y_true):
        with np.errstate(divide='ignore', invalid='ignore'):
            loss_result = -np.max(y_true*np.log(y_pred) + (1 - y_true)*np.log(1 - y_pred))
            return loss_result
        # return - np.mean(y_true*np.log(y_pred) - (1 - y_true)*(1 - y_pred))
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, eta=0.01, n_epochs=1000, tol=0.01, n_epochs_no_update=10):
        X_train = self.add_bias(X_train)  
        
        m, n = X_train.shape      
        self.weights = np.zeros(n)

        prev_loss = n_epochs
        no_update = 0

        if X_val is None and y_val is None:
            for epoch in range(n_epochs):
                # y_pred = self.forward(X_train)
                # Gradient
                # g = (X_train.T @ (y_pred - y_train))/m
                self.weights -= eta / m *  X_train.T @ (self.forward(X_train) - y_train)
                y_pred = self.forward(X_train)

                train_loss = self.loss(y_pred, y_train)
                self.train_losses.append(train_loss)
                train_acc = self.accuracy(y_pred, y_train)
                self.train_accuracies.append(train_acc)

                loss_update = prev_loss - train_loss
                if loss_update < tol:
                    no_update += 1
                else:
                    no_update = 0
                
                prev_loss = self.train_losses[-1]

                if no_update >= n_epochs_no_update:
                    self.epochs_trained = epoch + 1
                    # print(f"Break after {self.epochs}")
                    break

        else:
            X_val = self.add_bias(X_val)
            for epoch in range(n_epochs):
                self.weights -= eta / m *  X_train.T @ (self.forward(X_train) - y_train)

                y_out = self.forward(X_train)

                train_loss = self.loss(y_out, y_train)
                self.train_losses.append(train_loss)
                train_acc = self.accuracy(y_out, y_train)
                self.train_accuracies.append(train_acc)

                y_pred = self.forward(X_val)
                
                val_loss = self.loss(y_pred, y_val)
                self.val_losses.append(val_loss)

                val_acc = self.accuracy(y_pred, y_val)
                self.val_accuracies.append(val_acc)

                loss_update = prev_loss - val_loss
                if loss_update < tol:
                    no_update += 1
                else:
                    no_update = 0
                
                prev_loss = self.train_losses[-1]

                if no_update >= n_epochs_no_update:
                    self.epochs_trained = epoch + 1
                    # print(f"Break after {self.epochs}")
                    break


    def forward(self, X):
        return self.sigmoid(X @ self.weights)
    
    def predict(self, X, threshold=0.5):
        z = self.add_bias(X)
        score = self.forward(z)
        return (score>threshold).astype('int')
    
    def predict_proba(self, X):
        z = self.add_bias(X)
        return self.forward(z)
    
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
    def __init__(self, learning_rate, gradient, momentum = 0, optimizer = None, scheduler = None):
        self.learning_rate = learning_rate
        self.gradient = gradient
        self.momentum = momentum
        self.momentum_change = 0.0
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.theta = None
        self.n = None
        if self.scheduler is not None:
            print("Using learning rate scheduler, learning_rate argument is ignored")
    def _initialize_vars(self, X):
        self.theta = np.random.randn(X.shape[1], 1)
        self.n = X.shape[0]
    def _gd(self, grad, current_iter):
        if self.optimizer is None:
            update = self.learning_rate * grad + self.momentum * self.momentum_change
            self.momentum_change = update
        else:
            update = self.optimizer.calculate(self.learning_rate, grad, current_iter)

        return update

    def descend(self, X, y, n_iter=500):
        self._initialize_vars(X)
        for i in range(n_iter):
            if self.scheduler is not None:
                self.learning_rate = self.scheduler(i+1)
            grad = self.gradient(X, y, self.theta)
            update = self._gd(grad, i+1)
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
                if self.scheduler is not None:
                    self.learning_rate = self.scheduler(i * batch_size + j)
                random_index = batch_size * np.random.randint(n_batches)
                xi = xy[random_index:random_index+5, :-1]
                yi = xy[random_index:random_index+5, -1:]
                grad = (1/batch_size) * self.gradient(X, y, self.theta)
                update = self._gd(grad, current_iter = j+1)
                self.theta -= update


    