from typing import Callable
import numpy as np

from ..activations import Activation
from .base import Layer
from ..core import Tensor, initializers as initial


class Dense(Layer):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation: Activation = None,
        initializer: Callable = initial.random
    ) -> None:
        super().__init__()

        self.weights = Tensor(
            initializer(input_size, output_size),
            requires_grad=True
        )
        self.biases = Tensor(
            initial.zeros(output_size),
            requires_grad=True
        )
        self.activation = activation
        self.parameters = [self.weights, self.biases]

    def forward(self, input: Tensor) -> Tensor:
        self.input = input
        z = input @ self.weights + self.biases

        if self.activation:
            return self.activation.forward(z)

        return z

    def backward(self, grad_output: Tensor) -> Tensor:
        if self.activation:
            grad_output = self.activation.backward(grad_output)

        grad_weights = self.input.T @ grad_output
        grad_bias = np.sum(grad_output, axis=0, keepdims=True)
        self.gradients = [grad_weights, grad_bias]
        self.weights.grad = grad_weights
        self.biases.grad = grad_bias

        grad_input = grad_output @ self.weights.T
        return grad_input
