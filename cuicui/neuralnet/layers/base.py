from abc import ABC, abstractmethod
from typing import List, Optional
from ..core import Tensor


class Layer(ABC):
    def __init__(self) -> None:
        self.parameters: List[Tensor] = []
        self.gradients: Optional[Tensor] = []

    @abstractmethod
    def forward(self, input: Tensor) -> Tensor:
        """
        Computes the output of the layer for a given input.
        """
        pass

    @abstractmethod
    def backward(self, grad_output: Tensor) -> Tensor:
        pass

    def get_parameters(self) -> List[Tensor]:
        """
        Return parameters (weights and biases) of Layer.
        """
        return self.parameters

    def get_gradients(self) -> List[Tensor]:
        """
        Return gradients of Layer parameters.
        """
        return self.gradients

    def zero_grad(self) -> None:
        """
        Reset all Layer paramter gradients.
        """
        for i in range(len(self.gradients)):
            if self.gradients[i] is not None:
                self.gradients[i] = None
