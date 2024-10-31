import autograd.numpy as np

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