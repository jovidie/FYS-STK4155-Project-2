from os import access
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score

def set_plt_params(remove_grid=False):
    """Set parameters and use seaborn theme to plot."""
    sns.set_theme()
    if remove_grid:
        sns.set_style("whitegrid", {"axes.grid": False})
    params = {
        "font.family": "Serif",
        "font.serif": "Roman", 
        "text.usetex": True,
        "axes.titlesize": "large",
        "axes.labelsize": "large",
        "xtick.labelsize": "large",
        "ytick.labelsize": "large",
        "legend.fontsize": "medium", 
        "savefig.dpi": 300, 
        "axes.grid" : False
    }
    plt.rcParams.update(params)


def plot_heatmap(etas, lmbdas, acc, figname=None):
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
    lmbdas_, etas_ = np.meshgrid(lmbdas, etas)

    idx = np.where(acc == acc.max())

    fig, ax = plt.subplots()
    c = sns.color_palette("mako", as_cmap=True)
    cs = ax.contourf(lmbdas_, etas_, acc, levels=len(lmbdas), cmap=c)

    # Include point where optimal parameters are
    lmbda_opt = lmbdas[idx[1]]
    eta_opt = etas[idx[0]]
    ax.plot(lmbda_opt, eta_opt, "X", label="Optimal") # $\lambda = {lmbda_opt}$ $\eta = {eta_opt}$
    ax.legend(title=f"Accuracy = {acc.max():.4f}")

    fig.colorbar(cs, label="Accuracy")

    ax.set_xlabel(r"$Log_{10}(\lambda)$")
    ax.set_ylabel("Learning rate")

    if figname is not None:
        fig.savefig(f"latex/figures/{figname}.pdf", bbox_inches="tight")
        
    else:
        plt.show()


def plot_mse_r2(common, mse, r2, figname=None):
    """Function to show how to plot MSE and R2 score in one plot.
    
    Args:
        common (np.ndarray): common feature to plot
        mse (np.ndarray): MSE values
        r2 (np.ndarray): R2 score values
		figname (str): saves figure with the given name 
		
	Returns:
        None
    """
    c = sns.color_palette("mako", n_colors=2, as_cmap=False)

    fig, ax = plt.subplots(layout='constrained')
    ax2 = ax.twinx()
    ax.plot(common, mse, color=c[0],  label="MSE")
    ax2.plot(common, r2, color=c[1], label=r"R$^{2}$")

    ax2.grid(None)

    mse_lines, mse_labels = ax.get_legend_handles_labels()
    r2_lines, r2_labels = ax2.get_legend_handles_labels() 
    ax2.legend(mse_lines+r2_lines, mse_labels+r2_labels, loc="upper right")

    # Change Common to feature name to plot against eg. degree, epoch
    ax.set_xlabel("Common")
    ax.set_ylabel("MSE")
    ax2.set_ylabel(r"R$^{2}$")

    if figname is not None:
        fig.savefig(f"latex/figures/{figname}.pdf", bbox_inches="tight")
    
    else:
        plt.show()



def plot_mse(common, mse_test, mse_train=None, figname=None):
    """
    Args:
        common (np.ndarray): common feature to plot
        mse_test (np.ndarray): MSE test values
        mse_train (np.ndarray): MSE train values, default is None
		figname (str): saves figure with the given name 
		
	Returns:
        None
    """
    c = sns.color_palette("mako", n_colors=2, as_cmap=False)

    fig, ax = plt.subplots(layout='constrained')

    ax.plot(common, mse_test, color=c[0],  label=r"MSE$_{test}$")
    
    if mse_train is not None:
        ax.plot(common, mse_train, "--", color=c[0], label=r"MSE$_{train}$")

    ax.legend(loc="upper right")

    # Change Common to feature name to plot against eg. degree, epoch
    ax.set_xlabel("Common")
    ax.set_ylabel("MSE")

    if figname is not None:
        fig.savefig(f"latex/figures/{figname}.pdf", bbox_inches="tight")
        
    else:
        plt.show()


def plot_loss_acc(common, loss, acc, figname=None):
    """Function to show how to plot MSE and R2 score in one plot.
    
    Args:
        common (np.ndarray): common feature to plot
        mse (np.ndarray): MSE values
        r2 (np.ndarray): R2 score values
		figname (str): saves figure with the given name 
		
	Returns:
        None
    """
    c = sns.color_palette("mako", n_colors=2, as_cmap=False)

    fig, ax = plt.subplots(layout='constrained')
    ax2 = ax.twinx()
    ax.plot(common, loss, color=c[0],  label="Loss")
    ax2.plot(common, acc, color=c[1], label="Accuracy")

    ax2.grid(None)

    loss_lines, loss_labels = ax.get_legend_handles_labels()
    acc_lines, acc_labels = ax2.get_legend_handles_labels() 
    ax2.legend(loss_lines+acc_lines, loss_labels+acc_labels, loc="upper right")

    # Change Common to feature name to plot against eg. degree, epoch
    ax.set_xlabel("Common")
    ax.set_ylabel("Loss")
    ax2.set_ylabel("Accuracy")

    if figname is not None:
        fig.savefig(f"latex/figures/{figname}.pdf", bbox_inches="tight")
    
    else:
        plt.show()


def plot_confusion(y_pred, y_test, title, figname=None):
    acc = accuracy_score(y_test, y_pred)
    c = sns.color_palette("mako", as_cmap=True)
    conf1 = confusion_matrix(y_test, y_pred, labels = [0, 1])
    ConfusionMatrixDisplay(conf1).plot(cmap=c)
    plt.title(f"{title}, acc: {acc:.4f}")

    if figname is not None:
        plt.savefig(f"latex/figures/{figname}.pdf", bbox_inches="tight")
    
    else:
        plt.show()
