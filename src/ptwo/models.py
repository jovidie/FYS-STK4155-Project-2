import autograd.numpy as np
from sklearn.metrics import accuracy_score
from autograd import grad

class NeuralNetwork:
    """
    Neural Network model

    Args: 
    - network_input_size: number of data points, typically the first dimension of the design matrix
    - targets: the golden truth
    - layer_output_sizes = size of layers, number of layers is determined by len(layer_output sizes)
    - activation_funcs: a list of activation functions for the hidden and output layers
    - cost_function: cost function for the output, must be a function C(predicts, targets) that returns a single number
    - optimizer: an instance of an optimizer object Momentum, ADAM, AdaGrad or RMSProp (optional)
    - lmb: L2 regularization parameter (default 0)
    """
    def __init__(self, 
                 network_input_size, 
                 layer_output_sizes, 
                 activation_funcs, 
                 cost_function,
                 optimizer = None,
                 lmb = 0):
        self.network_input_size = network_input_size
        self.layer_output_sizes = layer_output_sizes
        self.activation_funcs = activation_funcs
        self.cost_function = cost_function
        self.optimizer = optimizer
        self.lmb = lmb
        self.create_layers_batch()

    def create_layers_batch(self):
        """
        Function that creates all the NN layers based on layer_output_sizes 
        Args:   
        input size of network: determines first layer size
        layer output size: Output sizes of the rest of the layers, where the last layer has to match target size
        returns nothing, saves layers as instance variable
        """
        self.layers = []
        i_size = self.network_input_size
        for layer_output_size in self.layer_output_sizes:
            W = np.random.randn(i_size, layer_output_size)
            b = np.random.randn(layer_output_size)
            self.layers.append((W, b))
            i_size = layer_output_size

    def _cost(self, x, targets, layers = None):
        if layers is None:
            layers = self.layers
        predictions = self.feed_forward_batch(x, layers)
        base_cost = self.cost_function(predictions, targets)
        
        # L2 regularization term
        l2_term = 0
        for W, b in layers:
            l2_term += np.sum(W**2)
        l2_term *= (self.lmb / 2.0)
        
        return base_cost + l2_term


    def feed_forward_batch(self, x, layers=None):
        """
        Function that transforms input data into predictions based on current weights and biases
        Args: 
        - x: input data to be transformed
        - layers: defaults to self.layers, necessary for automatic differentiation
        returns predictions from the output layer
        """
        if layers is None:
            layers = self.layers
        a = x
        for (W, b), activation_func in zip(layers, self.activation_funcs):
            z = a @ W + b 
            a = activation_func(z)
        return a
    
    def get_cost(self, inputs, targets):
        predictions = self.feed_forward_batch(inputs)
        return self.cost_function(predictions, targets)

    # TODO - fix proba
    def predict_proba(self, x):
        probs = self.feed_forward_batch(x)
        self.predictions = probs
        return np.argmax(probs, axis = 1)
    
    def predict(self, x):
        """
        """
        probs = self.feed_forward_batch(x)
        self.predictions = probs
        return np.argmax(probs, axis = 1)
    
    def accuracy(self, input, targets):
        """
        Calculate accuracy for a classification with one hot predictions. Feeds the data through
        the neural network and compares the output with the targets
        Args:
        - input: input data
        - targets: target data
        """
        predictions = self.feed_forward_batch(input)
        one_hot_predictions = np.zeros(predictions.shape)
        for i, prediction in enumerate(predictions):
            one_hot_predictions[i, np.argmax(prediction)] = 1
        self.prediction_accuracy = accuracy_score(one_hot_predictions, targets)
        return self.prediction_accuracy

    def _train(self, grad, learning_rate, current_iter, current_layer = None, current_var = None):
        if self.optimizer is None:
            update = learning_rate * grad
        else:
            if not self.optimizer.has_layers:
                self.optimizer.initialize_layers(self.layers)
            update = self.optimizer.calculate(learning_rate, grad, current_iter, current_layer, current_var)

        return update

    def train_network(self, train_input, train_targets, learning_rate=0.001, epochs=100, batch_size = None): #TODO add cost
        """
        This function performs the back-propagation step to adjust the weights and biases
        for a default of 100 epochs with a default learning rate of 0.001. If no batch size
        is specified (default) it will run regular gradient descent (GD), if batch size is specified
        it wil run stochastic gradient descent (SGD) with the specified batch size.
        Args: 
        - train_input: the input variable x we use to predict y, should be a selection of data
        - train_targets: the matching golden truth to the train_input
        - cost: a selected cost function C(predict, target)
        - learning rate: determines the stepsize we take towards reaching the optimal W and b
        - epochs: number of iterations in one training cycle to reach optimal W and b
        - batch_size: batch size to use for SGD
        """
        if batch_size is None:
            self._train_network_gd(train_input, train_targets, learning_rate, epochs)
        else:
            self._train_network_sgd(train_input, train_targets, learning_rate, epochs, batch_size)
    
    def _train_network_gd(self, train_input, train_targets, learning_rate, epochs):
        self.train_input = train_input
        self.train_targets = train_targets
        gradient_func = grad(self._cost, 2)
        for i in range(epochs):
            layers_grad = gradient_func(train_input, train_targets, self.layers)
            j=0
            for (W, b), (W_g, b_g) in zip(self.layers, layers_grad):
                W -= self._train(W_g + self.lmb, learning_rate, i+1, current_layer=j, current_var=0)
                b -= self._train(b_g, learning_rate, i+1, current_layer=j, current_var=1)
                j+=1

    def _train_network_sgd(self, train_input, train_targets, learning_rate, epochs, batch_size):
        self.train_input = train_input
        self.train_targets = train_targets
        n = train_input.shape[0]
        n_output = train_targets.shape[1]
        n_batches = int(n / batch_size)
        gradient_func = grad(self._cost, 2)
        xy = np.column_stack([train_input,train_targets]) # for shuffling x and y together

        for i in range(epochs):
            if self.optimizer is not None:
                self.optimizer.reset(self.layers)
            np.random.shuffle(xy)
            for _ in range(n_batches):
                random_index = batch_size * np.random.randint(n_batches)
                xi = xy[random_index:random_index+batch_size, :-n_output]
                yi = xy[random_index:random_index+batch_size, -n_output:]

            layers_grad = gradient_func(xi, yi, self.layers)
            j=0
            for (W, b), (W_g, b_g) in zip(self.layers, layers_grad):
                W -= self._train(W_g + self.lmb, learning_rate, i+1, current_layer=j, current_var=0)
                b -= self._train(b_g, learning_rate, i+1, current_layer=j, current_var=1)
                j+=1

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
    def __init__(self, learning_rate, gradient, optimizer = None, scheduler = None):
        self.learning_rate = learning_rate
        self.gradient = gradient
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
            update = self.learning_rate * grad
        else:
            update = self.optimizer.calculate(self.learning_rate, grad, current_iter)

        return update
    def descend(self, X, y, epochs=100, batch_size=None):
            
        if batch_size is None:
            self._descend_gd(X, y, epochs)
        else:
            self._descend_sgd(X, y, epochs, batch_size)

    def _descend_gd(self, X, y, epochs):
        self._initialize_vars(X)
        for i in range(epochs):
            if self.scheduler is not None:
                self.learning_rate = self.scheduler(i+1)
            grad = self.gradient(X, y, self.theta)
            update = self._gd(grad, i+1)
            self.theta -= update

    def _descend_sgd(self, X, y, epochs, batch_size):
        self._initialize_vars(X)
        n_batches = int(self.n / batch_size)
        xy = np.column_stack([X,y]) # for shuffling x and y together
        for i in range(epochs):
            if self.optimizer is not None:
                self.optimizer.reset()
            np.random.shuffle(xy)
            for j in range(n_batches):
                if self.scheduler is not None:
                    self.learning_rate = self.scheduler(i * batch_size + j)
                random_index = batch_size * np.random.randint(n_batches)
                xi = xy[random_index:random_index+batch_size, :-1]
                yi = xy[random_index:random_index+batch_size, -1:]
                grad = (1/batch_size) * self.gradient(xi, yi, self.theta)
                update = self._gd(grad, current_iter = j+1)
                self.theta -= update


    