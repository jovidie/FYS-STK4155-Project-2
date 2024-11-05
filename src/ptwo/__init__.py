#!/usr/bin/env python
# -*- coding: utf-8 -*-


import importlib.metadata

# Example
from .models import NeuralNetwork, LogisticRegression, GradientDescent
from .optimizers import ADAM, AdaGrad, RMSProp, lr_scheduler
from .gradients import grad_OLS, grad_ridge
from .activators import ReLU, leaky_ReLU, softmax, softmax_vec, sigmoid
from .costfuns import mse, cross_entropy, binary_cross_entropy
from .utils import set_plt_params, calculate_polynomial, franke_function, GD_lambda_mse, eta_lambda_grid, lambda_lr_heatmap

__version__ = importlib.metadata.version(__package__)

__all__ = [
    "NeuralNetwork",
    "LogisticRegression", 
    "GradientDescent", 
    "ADAM",
    "AdaGrad",
    "RMSProp",
    "lr_scheduler",
    "grad_OLS",
    "grad_ridge",
    "ReLU",
    "leaky_ReLU",
    "softmax",
    "softmax_vec",
    "sigmoid",
    "mse",
    "cross_entropy",
    "binary_cross_entropy",
    "set_plt_params",
    "calculate_polynomial",
    "franke_function",
    "GD_lambda_mse",
    "eta_lambda_grid",
    "lambda_lr_heatmap"
]