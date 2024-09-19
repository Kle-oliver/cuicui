"""
Activation functions are important because they introduce non-linearity into
the neural network. This makes the model adaptive and capable of representing
complex data patterns. Activation functions determine weather the neuron's
output should be passed on to the next layer.
"""

from .base import Activation
from .sigmoid import Sigmoid

__all__ = ['Activation', 'Sigmoid']
