import numpy as np

from ..core import Tensor
from .base import Loss


class MSE(Loss):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, predicted: Tensor, target: Tensor) -> Tensor:
        # Stores entries for using in backpropagation
        self.predicted = predicted
        self.target = target

        self.loss = Tensor(
            np.mean((predicted - target)**2),
            requires_grad=False
        )
        return self.loss

    def backward(self) -> None:
        # Calculate gradient of MSE function
        grad = 2 * (self.predicted - self.target) / self.predicted.size
        self.predicted.backward(grad)
