from ..core import Tensor


class Loss:
    def forward(self, predicted: Tensor, target: Tensor) -> Tensor:
        raise NotImplementedError

    def backward(self) -> None:
        raise NotImplementedError
