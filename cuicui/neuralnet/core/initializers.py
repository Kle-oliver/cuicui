"""
Initializers are methods that we use to define
the initial values for layer parameters.
"""

import numpy as np

from .tensor import Tensor


def zeros(output_size: int) -> Tensor:
    """
    Initializes biases with zeros.
    """
    return Tensor(np.zeros((1, output_size)))


def random(
    input_size: int,
    output_sizes: int,
    scale: float = 0.01
) -> Tensor:
    """
    Initializes weights with normal distribution and multiplies by 'sacle'.
    """
    return Tensor(np.random.randn(input_size, output_sizes) * scale)


def xavier(input_size: int, output_size: int) -> Tensor:
    """
    Initializes weights with Xavier Initialization (Glorot Uniform).
    """
    limit = np.sqrt(6 / (input_size + output_size))
    return Tensor(np.random.uniform(-limit, limit, (input_size, output_size)))


def he(input_size: int, output_size: int) -> Tensor:
    """
    Initializes weights using He Initialization.
    """
    return Tensor(
        np.random.randn(input_size, output_size) * np.sqrt(2/input_size)
    )
