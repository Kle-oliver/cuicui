from typing import List

from ..core import Tensor


class Optimizer:
    def __init__(self, parameters: List[Tensor], learning_rate: float) -> None:
        self.parameters = parameters
        # Learning rate control the size of model parameters
        self.learning_rate = learning_rate

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for paramter in self.parameters:
            if paramter.grad is not None:
                paramter.grad = None
