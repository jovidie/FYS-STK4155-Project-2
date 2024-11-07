import autograd.numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from imageio.v2 import imread

from ptwo.utils import eta_lambda_grid, lambda_lr_heatmap
from ptwo.models import GradientDescent
from ptwo.optimizers import Momentum, ADAM, AdaGrad, RMSProp
from ptwo.gradients import grad_OLS, grad_ridge
from ptwo.costfuns import mse
from ptwo.plot import set_plt_params

set_plt_params()


terrain_full = imread('data/SRTM_data_Norway_2.tif')
# subset to a manageable size
terrain1 = terrain_full[1050:1250, 500:700]

x_1d = np.arange(terrain1.shape[1])
y_1d = np.arange(terrain1.shape[0])
# create grid
x_2d, y_2d = np.meshgrid(x_1d,y_1d)

# flatten the data and features
xy = np.column_stack((x_2d.flatten(), y_2d.flatten()))
max_poly = 10
X_feat = PolynomialFeatures(max_poly).fit_transform(xy)
X = X_feat[:, 1:]

y = terrain1.flatten()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
y_test = y_test[:,np.newaxis]
y_train = y_train[:,np.newaxis]

scalerX = StandardScaler(with_std = True)
X_train_scaled = scalerX.fit_transform(X_train)
X_test_scaled = scalerX.transform(X_test)

scalery = StandardScaler(with_std = True)
y_train_scaled = scalery.fit_transform(y_train)
y_test_scaled = scalery.transform(y_test)


# results from grid search applied to the terrain data
# even though grid search is technically performed below here

np.random.seed(2309148230)
learning_rate=1.78e-3
n_iter = 1000
lmb=1.78e-6
grad = grad_ridge(lmb)
#grad = grad_OLS()
gd = GradientDescent(learning_rate, grad,  optimizer = ADAM())
gd.descend(X_train_scaled, y_train_scaled, n_iter, batch_size = 32)
pred = X_test_scaled@gd.theta
print(mse(pred, y_test_scaled))

X_scaled = scalerX.fit_transform(X)
pred = X_scaled@gd.theta
y_predict = pred.reshape(200,200)
plt.imshow(y_predict, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig("examples/tests_even/figs/gradient-descent-terrain-map.pdf")

plt.show()

# GRID SEARCH
# TAKES VERY LONG TO RUN!

# np.random.seed(432787)
# learning_rates = np.logspace(-4, 1, 5)
# n_iter = 1000
# lmbs=np.logspace(-8,1, 5)

# mses = eta_lambda_grid(
#     X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled,
#     learning_rates, lmbs, n_iter, optimizer = ADAM()
# )
# lambda_lr_heatmap(mses, lmbs, learning_rates, filename="examples/tests_even/figs/terrain-gridsearch-gd.pdf")


# learning_rates = np.logspace(-4, 1, 5)
# n_iter = 200
# lmbs=np.logspace(-8,1, 5)
# batch_size=32

# mses = eta_lambda_grid(
#     X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled,
#     learning_rates, lmbs, n_iter, optimizer = ADAM(), batch_size = batch_size
# )
# lambda_lr_heatmap(mses, lmbs, learning_rates, filename="examples/tests_even/figs/terrain-gridsearch-sgd.pdf")