"""
An array and a tensor are essentially the same thing.
The difference lies in the context.
In Machine Learning, it is common to refer to arrays as tensors.
"""

from typing import Optional
import numpy as np


class Tensor(np.ndarray):
    def __new__(cls, input, requires_grad: bool = False):
        # Convert the input to a numpy array,
        # but the view is a Tensor class
        obj = np.asarray(input).view(cls)
        obj.requires_grad = requires_grad
        obj.grad = None
        obj._op = None  # This is the operation of Tensor, if there is
        return obj

    # def __getitem__(self, key):
    #     sliced_array = super(Tensor, self).__getitem__(key)
    #     return Tensor(sliced_array, requires_grad=self.requires_grad)

    def __add__(self, other: 'Tensor') -> 'Tensor':
        from .operations import Add

        op = Add()
        result = op.forward(self, other)
        result._op = op
        return result

    def __mul__(self, other: 'Tensor') -> 'Tensor':
        from .operations import Mul

        op = Mul()
        result = op.forward(self, other)
        result._op = op
        return result

    def backward(self, grad: Optional['Tensor'] = None) -> None:
        if self.grad is None:
            self.grad = np.zeros_like(self)

        if grad is None:
            grad = np.zeros_like(self)

        self.grad += grad

        if self._op is not None:
            self._op.backward(grad)

    def zero_grad(self):
        self.grad = None
