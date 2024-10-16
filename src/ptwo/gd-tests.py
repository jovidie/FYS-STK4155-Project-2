import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from utils import *
from GradientDescent import *

np.random.seed(8923)

# generate data
def data_function(x):
    return 4*x**3 + x**2 - 17*x + 48

n = 100
x = np.linspace(-2, 2, n)
f_x = data_function(x)
y = f_x + np.random.normal(0, 1, n)
y = y.reshape(-1,1)

# create design matrix of polynomials
# for now 3rd order reflecting data function
X = PolynomialFeatures(3).fit_transform(x.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

scalerx = StandardScaler()
X_train_scaled = scalerx.fit_transform(X_train)
X_test_scaled = scalerx.transform(X_test)

scalery = StandardScaler()
y_train_scaled = scalerx.fit_transform(y_train)
y_test_scaled = scalerx.transform(y_test)

# initialize parameters
eta = 0.09
n_iter = 200
grad = grad_OLS()

gd = GradientDescent(eta, grad)

gd.descend(X, y, n_iter = n_iter)
print("Regular", gd.theta, sep = "\n")

gamma = 0.3

gd_momentum = GradientDescent(eta, grad, momentum = True, momentum_gamma = 0.3)
gd_momentum.descend(X, y, n_iter = n_iter)
print("Momentum", gd_momentum.theta, sep = "\n")

eta = 2

adam = ADAM()
gd_ADAM = GradientDescent(eta, grad, adaptive = adam)
gd_ADAM.descend(X, y, n_iter = n_iter)
print("ADAM",gd_ADAM.theta, sep = "\n")

adagrad = AdaGrad()
gd_AdaGrad = GradientDescent(eta, grad, adaptive = adagrad)
gd_AdaGrad.descend(X, y, n_iter = n_iter)
print("AdaGrad", gd_AdaGrad.theta, sep = "\n")

rho = 0.99
rmsprop = RMSProp(rho = 0.99)
gd_RMSProp = GradientDescent(eta, grad, adaptive = rmsprop)
gd_RMSProp.descend(X, y, n_iter = n_iter)
print("RMSProp", gd_RMSProp.theta, sep = "\n")

# SGD
eta = 0.09
grad = grad_OLS()

M = 5
n_epochs = 50

gd = GradientDescent(eta, grad)

gd.descend_stochastic(X, y, n_epochs = n_epochs, batch_size = M)
print("Stochastic regular", gd.theta, sep = "\n")

adam = ADAM()
gd_ADAM = GradientDescent(eta, grad, adaptive = adam)
gd_ADAM.descend_stochastic(X, y, n_epochs = n_epochs, batch_size = M)
print("Stochastic ADAM",gd_ADAM.theta, sep = "\n")