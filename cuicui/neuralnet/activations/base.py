from ..core import Tensor, Operation


class Activation(Operation):
    def __init__(self) -> None:
        super().__init__()

    def _activation(self, x: Tensor):
        raise NotImplementedError

    def _activation_grad(self, x: Tensor):
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        # Stores entries for using in backpropagation
        self.inputs = [x]
        self.output = Tensor(
            self._activation(x),
            requires_grad=x.requires_grad
        )

        return self.output

    def backward(self, output_grad: Tensor) -> None:
        x = self.inputs[0]

        grad_input = output_grad * self._activation_grad(x)
        if x.requires_grad:
            x.backward(grad_input)

        return grad_input
