import numpy as np

class ADAM:
    def __init__(self, beta1=0.9, beta2=0.999, delta=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.delta = delta
        self.first_moment = 0.0
        self.second_moment = 0.0

    def reset(self):
        self.first_moment = 0.0
        self.second_moment = 0.0

    def calculate(self, learning_rate, grad, current_iter):
        self.first_moment = self.beta1*self.first_moment + (1-self.beta1)*grad
        self.second_moment = self.beta2*self.second_moment + (1-self.beta2)*grad*grad

        first_term = self.first_moment/(1.0-self.beta1**current_iter)
        second_term = self.second_moment/(1.0-self.beta2**current_iter)
        update = learning_rate*first_term/(np.sqrt(second_term)+self.delta)

        return update



class AdaGrad:
    def __init__(self, delta = 1e-8):
        self.delta = delta
        self.Giter = 0.0

    def reset(self):
        self.Giter = 0.0
    
    def calculate(self, learning_rate, grad, current_iter):
        self.Giter += grad*grad
        update = learning_rate * grad / (self.delta + np.sqrt(self.Giter))
        return update


class RMSProp:
    def __init__(self, rho, delta = 1e-8):
        self.rho = rho
        self.delta = delta
        self.Giter = 0.0

    def reset(self):
        self.Giter = 0.0
    
    def calculate(self, learning_rate, grad, current_iter):
        self.Giter = self.rho*self.Giter + (1-self.rho)*grad*grad
        update = learning_rate * grad / (self.delta+np.sqrt(self.Giter))
        return update

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


class GradientDescent:
    def __init__(self, learning_rate, gradient, momentum = False, momentum_gamma = None, adaptive = None):
        self.learning_rate = learning_rate
        self.gradient = gradient
        self.momentum = momentum
        self.momentum_gamma = momentum_gamma
        self.adaptive = adaptive
        self.theta = None
        self.n = None
        if self.momentum:
            if self.momentum_gamma is None:
                raise Exception("No gamma specified for momentum")
            self.momentum_change = 0.0
    def _initialize_vars(self, X):
        self.theta = np.random.randn(X.shape[1], 1)
        self.n = X.shape[0]

    def _gd(self, grad, X, y, current_iter):
        if self.adaptive is None:
            update = self.learning_rate * grad
            if self.momentum:
                update += self.momentum_gamma * self.momentum_change
                self.momentum_change = update
        else:
            update = self.adaptive.calculate(self.learning_rate, grad, current_iter)

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
            if self.adaptive is not None:
                self.adaptive.reset()
            np.random.shuffle(xy)
            for j in range(n_batches):
                random_index = batch_size * np.random.randint(n_batches)
                xi = xy[random_index:random_index+5, :-1]
                yi = xy[random_index:random_index+5, -1:]
                grad = (1/batch_size) * self.gradient(X, y, self.theta)
                update = self._gd(grad, xi, yi, current_iter = j+1)
                self.theta -= update

