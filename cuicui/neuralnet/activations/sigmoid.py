import numpy as np

from .base import Activation
from ..core import Tensor


class Sigmoid(Activation):
    def __init__(self) -> None:
        super().__init__()

    def _activation(self, x: Tensor):
        """
        The sigmoid formula is: f(x) = 1 / (1 + e^(-x))
        """
        return 1 / (1 + np.exp(-x))

    def _activation_grad(self, x: Tensor):
        sig = self._activation(x)
        return sig * (1 - sig)
