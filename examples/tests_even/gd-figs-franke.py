import autograd.numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from imageio import imread

from ptwo.utils import franke_function, GD_lambda_mse, eta_lambda_grid, lambda_lr_heatmap
from ptwo.models import GradientDescent
from ptwo.optimizers import Momentum, ADAM, AdaGrad, RMSProp
from ptwo.gradients import grad_OLS, grad_ridge
from ptwo.costfuns import mse

# Make data.
x1 = np.arange(0, 1, 0.05)
x2 = np.arange(0, 1, 0.05)
x1, x2 = np.meshgrid(x1,x2)

z = franke_function(x1, x2) + np.random.normal(0,0.1,x1.shape)

x1x2 = np.column_stack((x1.flatten(), x2.flatten()))
max_poly = 10
X_feat = PolynomialFeatures(max_poly).fit_transform(x1x2)
X = X_feat[:, 1:] # remove intercept
y = z.flatten()

np.random.seed(42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# reshape y to 2D array
y_test = y_test[:,np.newaxis]
y_train = y_train[:,np.newaxis]

# scale data
scalerX = StandardScaler(with_std=True)
X_train_scaled = scalerX.fit_transform(X_train)
X_test_scaled = scalerX.transform(X_test)

scalery = StandardScaler(with_std=True)
y_train_scaled = scalery.fit_transform(y_train)
y_test_scaled = scalery.transform(y_test)

## TEST DIFFERENT OPTIMIZERS AND LEARNING RATES ##
np.random.seed(3089750)
learning_rates = np.logspace(-8,1, 20)
optimizers = [Momentum, AdaGrad, ADAM, RMSProp]
n_iter = 2000
lmb=1e-4
grad = grad_ridge(lmb)

mses = np.zeros( (len(learning_rates), len(optimizers)) )
for j, optimizer in enumerate(optimizers):
    for i, learning_rate in enumerate(learning_rates):
        gd = GradientDescent(learning_rate, grad, optimizer=optimizer())
        gd.descend(X_train_scaled, y_train_scaled, n_iter)
        y_pred = X_test_scaled @ gd.theta
        mses[i,j] = mse(y_pred, y_test_scaled)

optimizer_names = ["Momentum", "AdaGrad", "ADAM", "RMSProp"]
for j in range(len(optimizers)):
    plt.plot(learning_rates, mses[:,j], label=optimizer_names[j])
plt.legend()
plt.xlabel(r"Learning rate $\eta$")
plt.ylabel("Test MSE")
plt.xscale("log")
plt.yscale("log")
plt.savefig("examples/tests_even/figs/Franke-learningrates-optimizers.pdf")
plt.show()


## GRID SEARCH ##

np.random.seed(865489724)
learning_rates = np.logspace(-3, 1, 8)
n_iter = 1000
lmbs=np.logspace(-8,1, 8)

mses = eta_lambda_grid(
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled,
    learning_rates, lmbs, n_iter, optimizer=ADAM()
)
lambda_lr_heatmap(mses, lmbs, learning_rates, filename="examples/tests_even/figs/Franke-grid-search-gd-adam.pdf")

np.random.seed(865489724)
learning_rates = np.logspace(-3, 1, 8)
n_iter = 200
lmbs=np.logspace(-8,1, 8)
batch_size=32


mses = eta_lambda_grid(
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled,
    learning_rates, lmbs, n_iter, optimizer=ADAM(), batch_size=batch_size
)
lambda_lr_heatmap(mses, lmbs, learning_rates, filename="examples/tests_even/figs/Franke-grid-search-sgd-adam.pdf")