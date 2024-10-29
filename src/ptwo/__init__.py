#!/usr/bin/env python
# -*- coding: utf-8 -*-
import importlib.metadata
# Example
from .models import NeuralNetwork, LogisticRegression, LogReg, GradientDescent
from .activators import sigmoid, ReLU
__version__ = importlib.metadata.version(__package__)

__all__ = [
    "NeuralNetwork",
    "LogisticRegression", 
    "GradientDescent", 
    "LogReg", 
    "sigmoid", 
    "ReLU"
]