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

    def adjust_grad_for_broadcasting(
        self,
        input_tensor: Tensor,
        grad_output: np.ndarray
    ) -> np.ndarray:
        """
        Ajusta o gradiente de saída para levar em conta o broadcasting que
        ocorreu durante a passagem direta.

        :param input_tensor: O tensor de entrada original (por exemplo, y).
        :param grad_output: O gradiente de saída em relação à
        operação (por exemplo, output_grad).
        :return: O gradiente ajustado que corresponde
        à forma do tensor de entrada.
        """
        grad = grad_output

        input_shape = input_tensor.shape
        grad_shape = grad.shape

        # Expand the input tensor form to match grad_output, if necessary
        if len(input_shape) < len(grad_shape):
            input_shape += (1,) * (len(grad_shape) - len(input_shape))

        # Determinar os eixos onde ocorreu o broadcasting
        axes_to_sum = tuple(
            axis for axis, (dim_input, dim_grad)
            in enumerate(zip(input_shape, grad_shape))
            if dim_input == 1 and dim_grad > 1
        )

        # Sum the broadcasted axis
        if axes_to_sum:
            grad = grad.sum(axis=axes_to_sum, keepdims=True)

        # Ensure that the ajusted gradient has the
        # same shape as the input tensor
        if grad.shape != input_tensor.shape:
            grad = grad.reshape(input_tensor.shape)

        return grad


class Add(Operation):
    def forward(self, x: Tensor, y: Tensor):
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
            grad_wrt_y = self.adjust_grad_for_broadcasting(y, output_grad)
            y.backward(grad_wrt_y)


class Mul(Operation):
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        self.inputs = [x, y]
        x_data = np.asarray(x)
        y_data = np.asarray(y)
        self.output = Tensor(
            x_data * y_data,
            requires_grad=x.requires_grad or y.requires_grad
        )

        return self.output

    def backward(self, output_grad: Tensor) -> None:
        x, y = self.inputs

        if x.requires_grad:
            grad_wrt_x = output_grad * y
            grad_wrt_x = self.adjust_grad_for_broadcasting(x, grad_wrt_x)

            x.backward(grad_wrt_x)

        if y.requires_grad:
            grad_wrt_y = output_grad + x
            grad_wrt_y = self.adjust_grad_for_broadcasting(y, grad_wrt_y)
            y.backward(grad_wrt_y)
