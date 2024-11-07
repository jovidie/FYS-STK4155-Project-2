import matplotlib.pyplot as plt
import autograd.numpy as np

from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

from ptwo.utils import calculate_polynomial
from ptwo.models import GradientDescent
from ptwo.optimizers import Momentum, ADAM, AdaGrad, RMSProp
from ptwo.gradients import grad_OLS
from ptwo.plot import set_plt_params

set_plt_params()
# generate data
np.random.seed(238947987)
n = 100
x = np.linspace(-3, 2, n)
poly_list = np.array([3, 3, 1, 4])
y = calculate_polynomial(x, *poly_list) 
y = y.reshape(-1,1)

# choose polynomial degree that matches design
X = PolynomialFeatures(len(poly_list) - 1).fit_transform(x.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# Use gradient descent to find parameters
learning_rate = 0.01
n_iter = 1000
grad = grad_OLS()

gd = GradientDescent(learning_rate, grad)

step_size = 10
n_steps = int(n_iter / step_size)
convergence = np.zeros((n_steps,X.shape[1]))
print(convergence.shape)
iters = np.zeros(n_steps)
for i in range(n_steps):
    gd.descend(X, y, epochs = step_size)
    new_coef = gd.theta
    convergence[i, :] = (poly_list.reshape(-1, 1) - new_coef).flatten()
    iters[i] = i*step_size

print(gd.theta)
for i in range(X.shape[1]):
    plt.plot(iters, convergence[:,i], label=fr"$\beta_{i}$")

plt.legend()
plt.xlabel("Number of iterations")
plt.ylabel("Difference from true value")
plt.axhline(0, linestyle = "--", color="grey")
plt.savefig("examples/tests_even/figs/gradient-descent-polynomial-convergence.pdf")
plt.show()

## Same but with momentum

learning_rate = 0.01
n_iter = 1000
grad = grad_OLS()
momentum_gamma = 0.8

gd = GradientDescent(learning_rate, grad, optimizer=Momentum(momentum_gamma))

step_size = 10
n_steps = int(n_iter / step_size)
convergence = np.zeros((n_steps,X.shape[1]))
print(convergence.shape)
iters = np.zeros(n_steps)
for i in range(n_steps):
    gd.descend(X, y, epochs = step_size)
    new_coef = gd.theta
    convergence[i, :] = (poly_list.reshape(-1, 1) - new_coef).flatten()
    iters[i] = i*step_size

print(gd.theta)

for i in range(X.shape[1]):
    plt.plot(iters, convergence[:,i], label=fr"$\beta_{i}$")

plt.legend()
plt.xlabel("Number of iterations")
plt.ylabel("Difference from true value")
plt.axhline(0, linestyle = "--", color="grey")
plt.savefig("examples/tests_even/figs/gradient-descent-momentum-polynomial-convergence.pdf")
plt.show()

## Same but ADAM

learning_rate = 1
n_iter = 1000
grad = grad_OLS()
momentum_gamma = 0.8

gd = GradientDescent(learning_rate, grad, optimizer=ADAM())

step_size = 10
n_steps = int(n_iter / step_size)
convergence = np.zeros((n_steps,X.shape[1]))
print(convergence.shape)
iters = np.zeros(n_steps)
for i in range(n_steps):
    gd.descend(X, y, epochs = step_size)
    new_coef = gd.theta
    convergence[i, :] = (poly_list.reshape(-1, 1) - new_coef).flatten()
    iters[i] = i*step_size

print(gd.theta)

for i in range(X.shape[1]):
    plt.plot(iters, convergence[:,i], label=fr"$\beta_{i}$")

plt.legend()
plt.xlabel("Number of iterations")
plt.ylabel("Difference from true value")
plt.axhline(0, linestyle = "--", color="grey")
plt.savefig("examples/tests_even/figs/gradient-descent-ADAM-polynomial-convergence.pdf")
plt.show()