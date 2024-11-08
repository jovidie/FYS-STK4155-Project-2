import numpy as np
from ptwo.plot import set_plt_params, plot_heatmap
from ptwo.utils import lambda_lr_heatmap
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from imageio.v2 import imread

from ptwo.activators import ReLU, sigmoid
from ptwo.models import NeuralNetwork
from ptwo.optimizers import Momentum, ADAM, AdaGrad, RMSProp
from ptwo.costfuns import mse
from ptwo.plot import set_plt_params

#set_plt_params()

# modify heatmap function

def plot_heatmap2(etas, lmbdas, acc, figname=None):
    """Plot MSE in a heatmap as a function of lambda and learning rate.
    
    Args:
        etas (np.ndarray): learning rate values
        lmbdas (np.ndarray): lambda values
        mse (np.ndarray): MSE values
		figname (str): saves figure with the given name 
		
	Returns:
        None
    """
    lmbdas = np.log10(lmbdas)
    etas = np.log10(etas)
    etas_, lmbdas_ = np.meshgrid(etas, lmbdas)

    idx = np.where(acc == acc.min())

    fig, ax = plt.subplots()
    c = sns.color_palette("mako", as_cmap=True)
    cs = ax.contourf(lmbdas_, etas_, acc, levels=len(lmbdas), cmap=c)

    # Include point where optimal parameters are
    lmbda_opt = lmbdas[idx[0]]
    eta_opt = etas[idx[1]]
    ax.plot(lmbda_opt, eta_opt, "X", label="Optimal") # $\lambda = {lmbda_opt}$ $\eta = {eta_opt}$
    ax.legend(title=f"MSE = {acc.min():.4f}")
    print(fr"Optimal $\lambda$: {"{0:.2e}".format(float(10**lmbda_opt))}", fr"Optimal $\eta$: {"{0:.2e}".format(float(10**eta_opt))}", fr"MSE: {round(acc.min(),4)}", sep="\n")
    fig.colorbar(cs, label="MSE")

    ax.set_xlabel(r"$Log_{10}(\lambda)$")
    ax.set_ylabel(r"$Log_{10}(\eta)$")

    if figname is not None:
        fig.savefig(f"latex/figures/{figname}.pdf", bbox_inches="tight")
        
    else:
        plt.show()







terrain_full = imread('data/SRTM_data_Norway_2.tif')
# subset to a manageable size
terrain1 = terrain_full[1050:1250, 500:700]

x_1d = np.arange(terrain1.shape[1])
y_1d = np.arange(terrain1.shape[0])
# create grid
x_2d, y_2d = np.meshgrid(x_1d,y_1d)

# flatten the data and features
X = np.column_stack((x_2d.flatten(), y_2d.flatten()))

y = np.asarray(terrain1.flatten())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
y_test = y_test[:,np.newaxis]
y_train = y_train[:,np.newaxis]

scalerX = StandardScaler(with_std = True)
X_train_scaled = scalerX.fit_transform(X_train)
X_test_scaled = scalerX.transform(X_test)

scalery = StandardScaler(with_std = True)
y_train_scaled = scalery.fit_transform(y_train)
y_test_scaled = scalery.transform(y_test)

plt.imshow(terrain1, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig("examples/tests_even/figs/terrain-map.pdf")
plt.show()


#set_plt_params()
print("params set")
learning_rates = np.load("examples/tests_even/data/learning_rates3.npy")
lmbs=np.load("examples/tests_even/data/lmbs3.npy")
mse_gd = np.load("examples/tests_even/data/mses-terrain-gd3-adam.npy")
mse_sgd = np.load("examples/tests_even/data/mses-terrain-sgd3-adam.npy")

#lambda_lr_heatmap(mse_gd, lmbs, learning_rates)
#lambda_lr_heatmap(mse_sgd, lmbs, learning_rates)
#plot_heatmap2(learning_rates, lmbs, mse_gd, figname="nn-grid-search-gd")
#plot_heatmap2(learning_rates, lmbs, mse_sgd, figname="nn-grid-search-sgd")

## Final test with optimal values
np.random.seed(65345)
input_size = X_train.shape[1]
layer_output_sizes = [20, 10, 1]
activation_funs = [sigmoid, ReLU, lambda x: x]
learning_rate=1.67e-02
lmb = 1e-9

nn = NeuralNetwork(input_size, layer_output_sizes, activation_funs, mse, optimizer=ADAM())
print(nn.get_cost(X_test_scaled, y_test_scaled))
nn.train_network(X_train_scaled, y_train_scaled, learning_rate, epochs=50, batch_size=32)
print(nn.get_cost(X_test_scaled, y_test_scaled))

X_scaled = scalerX.transform(X)
pred = nn.feed_forward_batch(X_scaled)
y_predict = pred.reshape(200,200)
plt.clf()
plt.imshow(y_predict, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig("examples/tests_even/figs/neural-network-terrain-map.pdf")
plt.show()