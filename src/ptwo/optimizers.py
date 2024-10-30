import numpy as np

class ADAM:
    def __init__(self, layers = None, beta1=0.9, beta2=0.999, delta=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.delta = delta
        if layers is None:
            self.first_moment = 0.0
            self.second_moment = 0.0
        else:
            self.first_moment = []
            self.second_moment = []
            for W, b in layers:
                Ws = np.zeros(W.shape)
                bs = np.zeros(b.shape)
                self.first_moment.append([Ws, bs])
                self.second_moment.append([Ws, bs])


    def reset(self):
        self.first_moment = 0.0
        self.second_moment = 0.0

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

## Learning rate schedulers ##

def lr_scheduler(t0=1,t1=10):
    def _step(iteration):
        return t0/(iteration+t1)
    return _step



