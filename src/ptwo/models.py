import autograd.numpy as np
from sklearn.metrics import accuracy_score
from autograd import grad, elementwise_grad
from sklearn.utils import resample
from sklearn.model_selection import KFold
from scipy.special import expit

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
                 lmb = 0, 
                 target_means = None, classification = False):
        self.network_input_size = network_input_size
        self.layer_output_sizes = layer_output_sizes
        self.activation_funcs = activation_funcs
        self.optimizer = optimizer
        self.lmb = lmb
        self.target_means = target_means
        self.create_layers_batch()
        self.cost_function = cost_function
        self.classification = classification

    def create_layers_batch(self):
        # https://cs.stackexchange.com/questions/88360/neural-network-shape-structure
        # Neural net should have a pyramid shape
        """
        Function that creates all the NN layers based on layer_output_sizes 
        Args:
        input size of network: determines first layer size
        layer output size: Output sizes of the rest of the layers, where the last layer has to match target size
        returns nothing, saves layers as instance variable
        """
        self.layers = []
        numb_layers = len(self.layer_output_sizes)
        i_size = self.network_input_size
        lay = 0
        func_names = [func.__name__.lower() for func in self.activation_funcs]
        for ind, layer_output_size in enumerate(self.layer_output_sizes):
            if "leaky_relu" == func_names[ind].lower() or "relu" == func_names[ind].lower() or "relu6" == func_names[ind].lower() : 
                # He-normal initalization for rectified linear units
                W = np.random.randn(i_size, layer_output_size) * np.sqrt(2/i_size)
            else:
                # Xavier/Glorot initialization for sigmoidal non-linear activation functions
                W = np.random.randn(i_size, layer_output_size) * np.sqrt(1/i_size)
            b = np.random.randn(layer_output_size)
            # bias initialized as 0
            b = np.zeros(layer_output_size)
            # https://stackoverflow.com/questions/44883861/initial-bias-values-for-a-neural-network
            if self.target_means != None and lay == numb_layers-1: 
                b = self.target_means
            self.layers.append((W, b))
            i_size = layer_output_size
            lay += 1

    def _cost(self, x, targets, layers = None):
        if layers is None:
            layers = self.layers
        predictions = self.feed_forward_batch(x, layers)
        with np.errstate(divide='ignore', invalid='ignore'):
            base_cost = self.cost_function(predictions, targets)
        
        #L2 regularization term
        l2_term = 0
        for W, b in layers:
            l2_term += np.sum(W**2)
        l2_term *= (self.lmb / 2.0)

        return base_cost  + l2_term

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
    
    def predict_proba(self, x):
        probs = self.feed_forward_batch(x)
        self.predictions = probs
        return np.argmax(probs, axis = 1)
    
    def predict(self, x):
        """
        """
        pred = self.feed_forward_batch(x)
        self.predictions = pred
        return pred
    
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
        prediction_accuracy = accuracy_score(one_hot_predictions, targets)
        return prediction_accuracy

    def _train(self, grad, learning_rate, current_iter, current_layer = None, current_var = None):
        if self.optimizer is None:
            #assert np.sum(grad) != 0, f"\nGradient is zero, exiting program\n\nGRADIENTS BELOW:\n{grad}\n\n see self._train() method in NN for assertion."
            return learning_rate * grad
        else:
            try:
                return self.optimizer.calculate(learning_rate, grad, current_iter, current_layer, current_var)
            except: 
                self.optimizer.initialize_layers(self.layers)
                return self.optimizer.calculate(learning_rate, grad, current_iter, current_layer, current_var)

    def train_network(self, train_input, train_targets, learning_rate=0.001, epochs=100, batch_size = None, verbose = False):
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
        self.cost_evolution = []
        self.accuracy_evolution = []
        if self.optimizer is not None:
            self.optimizer.initialize_layers(self.layers)
        if batch_size is None:
            self._train_network_gd(train_input, train_targets, learning_rate, epochs, verbose)
        else:
            self._train_network_sgd(train_input, train_targets, learning_rate, epochs, batch_size, verbose)
        print("FINISHED TRAINING")

    def kfold_train(self, train_input, train_targets, learning_rate=0.001, epochs=100, batch_size = None, verbose = False, k = 5):
        kfold = KFold(n_splits=k)
        best_accuracy = 0

        for i, (train_ind, test_ind) in enumerate(kfold.split(train_input)):
            print("Fold:", i)
            self.create_layers_batch()
            self.train_network(train_input[train_ind], train_targets[train_ind], learning_rate, epochs, batch_size, verbose)
            if self.accuracy_evolution[-1] > best_accuracy:
                best_accuracy = self.accuracy(train_input[test_ind], train_targets[test_ind])
                keep_layers = self.layers
        print(f"Best accuracy on validation data achieved at {best_accuracy}, initializing saved weights from this run in NN")
        self.layers = keep_layers
    
    def _train_network_gd(self, train_input, train_targets, learning_rate, epochs, verbose):
        self.train_input = train_input
        self.train_targets = train_targets
        gradient_func = grad(self._cost, 2)
        for i in range(epochs):
            layers_grad = gradient_func(train_input, train_targets, self.layers)
            j = 0
            if i % 10 == 0 and verbose: #printer ut info pr. tiende epoke
                print("EPOCH:", i)
                print("COST FUNCTION:", self.get_cost(train_input, train_targets))
                self.cost_evolution.append(self.get_cost(train_input, train_targets))
                if self.classification: 
                    self.accuracy_evolution.append(self.accuracy(train_input, train_targets))
                
            for (W, b), (W_g, b_g) in zip(self.layers, layers_grad):
                W -= self._train(W_g + self.lmb, learning_rate, i + 1, current_layer = j, current_var = 0)
                b -= self._train(b_g, learning_rate, i + 1, current_layer = j, current_var = 1)
                j+=1
            

    def _train_network_sgd(self, train_input, train_targets, learning_rate, epochs, batch_size, verbose):
        self.train_input = train_input
        self.train_targets = train_targets
        input_rows = train_input.shape[0]
        n_batches = int(input_rows / batch_size)
        gradient_func = grad(self._cost, 2)
        inds = np.arange(input_rows)
        rng = np.random.default_rng()
        # epochs 
        i = 0 
        convergence_not_reached = True
        while i < epochs and convergence_not_reached: 

            # printing data as we go:
            if i % 10 == 0 and verbose: #printer ut info pr. x-te epoke
                cost = self.get_cost(train_input, train_targets)
                print("EPOCH:", i)
                print("COST FUNCTION:", cost)
                self.cost_evolution.append(self.get_cost(train_input, train_targets))
                if self.classification: 
                    self.accuracy_evolution.append(self.accuracy(train_input, train_targets))

            # what is this?
            #if self.optimizer is not None:
            #    self.optimizer.reset(self.layers)

            #splitting into batches: 
            rng.shuffle(inds) # shuffling rows of input
            index_batches = np.array_split(inds, n_batches) # splitting up random indexes into even-ish batches

            for indexes in index_batches: 
                x_batch = train_input[indexes, : ]
                y_batch = train_targets[indexes, :]
                layers_grad = gradient_func(x_batch, y_batch, self.layers)
                j = 0
                for (W, b), (W_g, b_g) in zip(self.layers, layers_grad):
                    W -= self._train((1/batch_size)*(W_g + self.lmb), learning_rate, i + 1, current_layer = j, current_var = 0)
                    b -= self._train((1/batch_size)*b_g, learning_rate, i + 1, current_layer = j, current_var = 1)
                    j += 1
                    
                #if cost < 10**-5:
                #    convergence_not_reached = False
                #    print("Converged at epoch", i)
                #    break
            
            #increase epoch
            i += 1

# Not necessary with both, condence classes into one
class LogisticRegression:
    """Logistic regression model, fit data using either gradient or stochastic
    gradient descent method. Predicts target probability using the sigmoid function,
    and target class using a threshold."""
    def __init__(self, lmbda=0.0, bias=1):
        self._lmbda = lmbda
        self._bias = bias
        self._beta = None
        self._train_losses = []
        self._train_accuracies = []
        self._epochs_trained = None

    @property
    def epochs(self):
        return self._epochs_trained
    
    @epochs.setter
    def epochs(self, epoch):
        self._epochs_trained = epoch

    @property
    def get_loss(self):
        return self._train_losses
    
    @property
    def get_accuracies(self):
        return self._train_accuracies
    
    def _init_params(self, X):
        self._n_data, self._n_features = X.shape 
        self._beta = np.random.randn(self._n_features)

    def _add_bias(self, X):
        N = X.shape[0]
        biases = np.ones((N, 1)) * self._bias
        return np.concatenate((biases, X), axis  = 1) 

    def _scheduler(self, t):
        return self._batch_size/(t + self._n_epochs)

    def _gd(self, X, y, n_epochs, tol, n_epochs_no_update):
        """Gradient descent solver method, to fit beta param.
        
        Args:
            X (np.ndarray): array of input data
            y (np.ndarray): array of target data
            n_epochs (int): number of iterations of training to fit beta

        Returns:
            None
        """
        prev_loss = n_epochs
        no_update = 0
        for epoch in range(n_epochs):
            y_pred = self.predict(X)
            grad_est = self.gradient(X, y, y_pred) 
            self._beta -= self._eta * grad_est
            # Evaluate 
            train_loss = np.mean(grad_est)
            self._train_losses.append(train_loss)
            train_acc = accuracy_score(y, y_pred)
            self._train_accuracies.append(train_acc)

            loss_update = prev_loss - train_loss
            if loss_update < tol:
                no_update += 1
            else:
                no_update = 0

            if no_update >= n_epochs_no_update: 
                self.epochs_trained = epoch + 1
                break

    def _sgd(self, X, y, n_epochs, batch_size):
        """Stochastic gradient descent solver method, to fit beta param.
        
        Args:
            X (np.ndarray): array of input data
            y (np.ndarray): array of target data
            n_epochs (int): number of iterations of training to fit beta
            batch_size (int): split data into batches of size 

        Returns:
            None
        """
        self._n_epochs = n_epochs
        self._batch_size = batch_size
        n_batches = int(self._n_data / batch_size)
        # xy = np.column_stack([X,y]) 

        for epoch in range(n_epochs):
            for i in range(n_batches):
                idx = batch_size*np.random.randint(n_batches)
                xi = X[idx:idx+batch_size]
                yi = y[idx:idx+batch_size]
                y_pred = self.predict(xi)
                grad = self.gradient(xi, yi, y_pred)
                # self._eta = self._optimizer(grad)
                self._eta = self._scheduler(epoch*n_batches+i)
                self._beta -= self._eta*grad 
                # self._optimizer.reset()


    def sigmoid(self, X):
        # Handles overflow in exp using scipy expit
        # return 1.0 / (1 + np.exp(-X))
        return 1.0 / (1 + expit(-X))

    def gradient(self, X, y, y_pred):
        return X.T @ (self.sigmoid(y_pred - y)) #/ self._n_data
    
    def forward(self, X):
        """Transform input data using the sigmoid function.

        Args: 
            X (np.ndarray): input data to be transformed
        
        Returns:
            np.ndarray of transformed values
        """
        a = X @ self._beta
        return self.sigmoid(a)
    
    def accuracy(self, y_pred, y_true, threshold=0.5):
        y_cat = (y_pred > threshold).astype('int')
        return np.mean(y_cat == y_true)
    
    def cost(self, y_true, y_pred):
        prob = y_pred*y_true + (1 - y_pred) * (1 - y_true)
        return - np.sum(np.log(prob))
        # with np.errstate(divide='ignore', invalid='ignore'):
        #     result = - np.mean(y_true*np.log(y_pred) + (1 - y_true)*np.log(1 - y_pred))
        #     return result
        # y_pred = self.predict_bias(X)
        # return - np.mean(y * np.log(y_pred + 1e-15) + (1 - y) * np.log(1 - y_pred + 1e-15))
        
    def cost_reg(self, y_pred, y_true):
        n = len(y_true)
        with np.errstate(divide='ignore', invalid='ignore'):
            term = - np.sum(y_true*np.log(y_pred) + (1 - y_true)*np.log(1 - y_pred)) / n
            l2 = self._lmbda * np.sum(self._beta*self._beta) / n
            return term + l2
    
    def fit(self, X_train, y_train, batch_size=None, optimizer=None, eta=0.01, n_epochs=1000, tol=0.001, n_epochs_no_update=10):
        """Train the model using either the gradient descent or stochastic 
        gradient descent method.
        
        Args:
            X_train (np.ndarray): array of training input data
            y_train (np.ndarray): array of training target data
            batch_size (int): split data into batches of size, train using sgd
            eta (float): learning rate
            n_epochs (int): number of iterations of training to fit beta

        Returns:
            None
        """
        self._eta = eta
        if optimizer is None:
            self._optimizer = self._scheduler
        else:
            self._optimizer = optimizer

        if self._beta is None:
            self._init_params(X_train) 
        
        if batch_size is None:
            self._gd(X_train, y_train, n_epochs, tol, n_epochs_no_update)
        
        else:
            self._sgd(X_train, y_train, n_epochs, batch_size)

    
    def predict(self, X, threshold=0.5):
        score = self.forward(X)
        self._y_pred = (score>threshold).astype('int')
        return self._y_pred
    
    def predict_bias(self, X, threshold=0.5):
        # X = self._add_bias(X)
        score = self.forward(X)
        self._y_pred = (score>threshold).astype('int')
        return self._y_pred
    

    def predict_proba(self, X):
        # X = self._add_bias(X)
        return self.forward(X)
    

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
        if self.theta is None:
            self._initialize_vars(X)
        if batch_size is None:
            self._descend_gd(X, y, epochs)
        else:
            self._descend_sgd(X, y, epochs, batch_size)

    def _descend_gd(self, X, y, epochs):
        for i in range(epochs):
            if self.scheduler is not None:
                self.learning_rate = self.scheduler(i+1)
            grad = self.gradient(X, y, self.theta)
            update = self._gd(grad, i+1)
            self.theta -= update

    def _descend_sgd(self, X, y, epochs, batch_size):
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


    