"""
Optimization functions are responsible for interatively adjusting the model's
weights and biases to minimize the loss functions, thereby improving the
model's perfomance.
"""

from .base import Optimizer
from .sgd import SGD

__all__ = ['Optimizer', 'SGD']
