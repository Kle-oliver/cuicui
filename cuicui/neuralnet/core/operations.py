"""
This module is responsible for modularizing the mathematical operations with
forward and backwards methods.
"""

import numpy as np
from typing import List

from .tensor import Tensor


class Operation:
    def __init__(self) -> None:
        self.inputs: List[Tensor] = []
        self.output: Tensor = None

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        raise NotImplementedError

    def backward(self, output_grad: Tensor) -> None:
        raise NotImplementedError


class Add(Operation):
    def forward(self, x: Tensor, y: Tensor):
        print(f"x type: {type(x)}, y type: {type(y)}")
        self.inputs = [x, y]
        self.output = Tensor(
            np.add(x, y),
            requires_grad=x.requires_grad or y.requires_grad
        )
        return self.output

    def backward(self, output_grad: Tensor) -> None:
        x, y = self.inputs
        if x.requires_grad:
            x.backward(output_grad)
        if y.requires_grad:
            y.backward(output_grad)


class Mul(Operation):
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        self.inputs = [x, y]
        self.output = Tensor(
            x * y,
            requires_grad=x.requires_grad or y.requires_grad
        )

        return self.output

    def backward(self, output_grad: Tensor) -> None:
        x, y = self.inputs
        if x.requires_grad:
            x.backward(output_grad * y)
        if y.requires_grad:
            y.backward(output_grad * x)
