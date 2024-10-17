#!/usr/bin/env python
# -*- coding: utf-8 -*-


import importlib.metadata

# Example
from .models import NeuralNetwork, LogisticRegression, GradientDescent

__version__ = importlib.metadata.version(__package__)

__all__ = [
    "NeuralNetwork",
    "LogisticRegression", 
    "GradientDescent", 
]