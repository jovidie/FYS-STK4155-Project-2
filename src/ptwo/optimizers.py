import numpy as np

class Momentum:
    def __init__(self, gamma=0.3, layers = None):
        self.gamma = gamma
        if layers is None:
            self.change = 0.0
            self.has_layers = False
    
    def initialize_layers(self, layers):
        self.change = []
        for W, b in layers:
            Ws = np.zeros(W.shape)
            bs = np.zeros(b.shape)
            self.change.append([Ws, bs])
        self.has_layers = True
    def reset(self, layers = None):
        if layers is None:
            self.change = 0
        else:
            self.initialize_layers(layers)
            
    def calculate(self, learning_rate, grad, current_iter, current_layer = None, current_var = None):
        if current_layer is None:
            update_term =  self.gamma * self.change
            update = learning_rate * grad + update_term
            self.change = update
        else:
            current_change = self.change[current_layer][current_var]
            update_term = self.gamma * current_change
            update = learning_rate * grad + update_term
            self.change[current_layer][current_var] = update
        return update


class ADAM:
    def __init__(self, layers = None, beta1=0.9, beta2=0.999, delta=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.delta = delta
        if layers is None:
            self.first_moment = 0.0
            self.second_moment = 0.0
            self.has_layers = False
        else:
            self.initialize_layers(layers)

    def initialize_layers(self, layers):
        self.first_moment = []
        self.second_moment = []
        for W, b in layers:
            Ws = np.zeros(W.shape)
            bs = np.zeros(b.shape)
            self.first_moment.append([Ws, bs])
            self.second_moment.append([Ws, bs])
        self.has_layers = True

    def reset(self, layers = None):
        if layers is None:
            self.first_moment = 0.0
            self.second_moment = 0.0
        else:
            self.initialize_layers(layers)

    def calculate(self, learning_rate, grad, current_iter, current_layer = None, current_var = None):
        if current_layer is None:
            first_moment = self.beta1*self.first_moment + (1-self.beta1)*grad
            second_moment = self.beta2*self.second_moment + (1-self.beta2)*grad*grad
            self.first_moment = first_moment
            self.second_moment = second_moment
        else:
            moment1 = self.first_moment[current_layer][current_var]
            moment2 = self.second_moment[current_layer][current_var]
            first_moment = self.beta1*moment1 + (1-self.beta1)*grad
            second_moment = self.beta2*moment2 + (1-self.beta2)*grad*grad
            self.first_moment[current_layer][current_var] = first_moment
            self.second_moment[current_layer][current_var] = second_moment

        first_term = first_moment/(1.0-self.beta1**current_iter)
        second_term = second_moment/(1.0-self.beta2**current_iter)
        update = learning_rate*first_term/(np.sqrt(second_term)+self.delta)
            

        return update



class AdaGrad:
    def __init__(self, layers = None, delta = 1e-8):
        self.delta = delta

        if layers is None:
            self.Giter = 0.0
            self.has_layers = False
        else:
            self.initialize_layers()

    def initialize_layers(self, layers):
        self.Giter = []
        for W, b in layers:
            Ws = np.zeros(W.shape)
            bs = np.zeros(b.shape)
            self.Giter.append([Ws, bs])
        self.has_layers = True
    
    def reset(self, layers = None):
        if layers is None:
            self.Giter = 0.0
        else:
            self.initialize_layers()
    
    def calculate(self, learning_rate, grad, current_iter, current_layer = None, current_var = None):
        if current_layer is None:
            Giter = self.Giter + grad*grad
            self.Giter = Giter
        else:
            current_Giter = self.Giter[current_layer][current_var]
            Giter = current_Giter + grad*grad
            self.Giter[current_layer][current_var] = Giter

        update = learning_rate * grad / (self.delta + np.sqrt(Giter))
        return update


class RMSProp:
    def __init__(self, rho=0.99, layers = None, delta = 1e-8):
        self.rho = rho
        self.delta = delta
        if layers is None:
            self.Giter = 0.0
            self.has_layers = False
        else:
            self.initialize_layers(layers)
    
    def initialize_layers(self, layers):
        self.Giter = []
        for W, b in layers:
            Ws = np.zeros(W.shape)
            bs = np.zeros(b.shape)
            self.Giter.append([Ws, bs])
        self.has_layers = True
    
    def reset(self):
        self.Giter = 0.0
    
    def calculate(self, learning_rate, grad, current_iter, current_layer = None, current_var = None):
        if current_layer is None:
            Giter = self.rho*self.Giter + (1-self.rho)*grad*grad
            self.Giter = Giter
        else:
            current_Giter = self.Giter[current_layer][current_var]
            Giter = self.rho*current_Giter + (1-self.rho)*grad*grad
            self.Giter[current_layer][current_var] = Giter

        update = learning_rate * grad / (self.delta+np.sqrt(Giter))
        return update

## Learning rate schedulers ##

def lr_scheduler(t0=1,t1=10):
    def _step(iteration):
        return t0/(iteration+t1)
    return _step



