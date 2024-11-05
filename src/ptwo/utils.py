import autograd.numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def set_plt_params():
    """Set parameters and use seaborn theme to plot."""
    sns.set_theme()
    params = {
        "font.family": "Serif",
        "font.serif": "Roman", 
        "text.usetex": True,
        "axes.titlesize": "large",
        "axes.labelsize": "large",
        "xtick.labelsize": "large",
        "ytick.labelsize": "large",
        "legend.fontsize": "medium", 
        "savefig.dpi": 300
    }
    plt.rcParams.update(params)


def calculate_polynomial(x, *args):
    """
    Calculate an arbitrary polynomial expression
    Args:
    - x: input value
    - *args: values for each polynomial degree:
    Returns: 
        args[0] + args[1]*x**1 + args[2]*x**2 + ... + args[n]*x**n
    Examples:
        x = np.random.randn(10)
        print(calculate_polynomial(x, 10, 2, -5))
        # or with a list:
        arglist = [10, 2, -5]
        print(calculate_polynomial(x, *arglist))
    """

    y = 0
    for i in range(len(args)):
        y += args[i] * x**i
    return y

def franke_function(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4