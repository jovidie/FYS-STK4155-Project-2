#!/usr/bin/env python
# -*- coding: utf-8 -*-


import importlib.metadata

# Example
from .models import NeuralNetwork, LogisticRegression, LogReg, GradientDescent
from .optimizers import ADAM, AdaGrad, RMSProp, lr_scheduler
from .gradients import grad_OLS, grad_ridge
from .activators import ReLU, leaky_ReLU, softmax, softmax_vec, sigmoid
from .costfuns import mse, cross_entropy, binary_cross_entropy
from .utils import calculate_polynomial, FrankeFunction

__version__ = importlib.metadata.version(__package__)

__all__ = [
    "NeuralNetwork",
    "LogisticRegression", 
    "GradientDescent", 
    "LogReg", 
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
    "calculate_polynomial",
    "franke_function"
]