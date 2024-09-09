from typing import List

from ..core import Tensor


class Optimizer:
    def __init__(self, paramters: List[Tensor], learning_rate: float) -> None:
        self.paramters = paramters
        # Learning rate control the size of model paramters
        self.learning_rate = learning_rate

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for paramter in self.paramters:
            if paramter.grad is not None:
                paramter.grad = None
