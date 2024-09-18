from typing import List
from cuicui.neuralnet.core import Tensor
from .base import Optimizer


class SGD(Optimizer):
    def __init__(
        self,
        parameters: List[Tensor] = [],
        learning_rate: float = 0.01
    ) -> None:
        super().__init__(parameters, learning_rate)

    def step(self):
        for paramter in self.parameters:
            if paramter.requires_grad:
                paramter -= self.learning_rate * paramter.grad
