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

    def __array_finalize__(self, obj) -> None:
        # Changing __array_finalize__ is important because, whenever we
        # perform operations like slicing, NumPy creates a new array for
        # optimization purposes. Therefore, we need to ensure that
        # our attributes persist

        if obj is None:
            return

        self.requires_grad = getattr(obj, 'requires_grad', False)
        self.grad = getattr(obj, 'grad', None)
        self._op = getattr(obj, '_op', None)

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
            grad = np.ones_like(self)

        self.grad += grad

        if self._op is not None:
            self._op.backward(grad)

    def zero_grad(self):
        self.grad = None
