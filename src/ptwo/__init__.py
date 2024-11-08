#!/usr/bin/env python
# -*- coding: utf-8 -*-


import importlib.metadata

# Example
from .models import NeuralNetwork, LogisticRegression, GradientDescent
from .optimizers import ADAM, AdaGrad, RMSProp, lr_scheduler
from .gradients import grad_OLS, grad_ridge
from .activators import ReLU, leaky_ReLU, softmax, softmax_vec, sigmoid
from .costfuns import mse, cross_entropy, binary_cross_entropy
from .utils import calculate_polynomial, franke_function, GD_lambda_mse, eta_lambda_grid, lambda_lr_heatmap, preprocess_cancer_data
from .plot import set_plt_params, plot_heatmap, plot_mse, plot_mse_r2, plot_loss_acc, plot_confusion

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
    "lambda_lr_heatmap", 
    "preprocess_cancer_data",
    "set_plt_params",
    "plot_heatmap",
    "plot_mse",
    "plot_mse_r2", 
    "plot_loss_acc",
    "plot_confusion"
]