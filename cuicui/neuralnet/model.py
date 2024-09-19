import h5py
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np

from .core import Tensor
from .layers import Layer
from .losses import Loss
from .optimizers import Optimizer


class Model:
    def __init__(self, layers: List[Layer]) -> None:
        """
        Initialize the model with a layer list.

        :param layers: List with layer instances (Dense, Conv2D, etc)
        """

        self.layers = layers
        self.loss: Loss
        self.optimizer: Optimizer
        self.metrics: Dict[str: Callable]

    def compile(
        self,
        loss: Loss,
        optimizer: Optimizer,
        metrics: List[Callable]
    ) -> None:
        """
        Model compile, define loss and optimizer function.

        :param loss: Function loss for use.
        :param optimizer: Function optimizer for use.
        :param metrics: List of function metrics loss for
        performance monitoring.
        """
        optimizer.parameters = self.get_parameters()

        self.loss = loss
        self.optimizer = optimizer
        self.metrics = {metric.__name__: metric for metric in metrics}

    def forward(self, x: Tensor) -> Tensor:
        """
        Execute forward propagation through all avaible layers of the model.

        :param x: The input tensor.
        :return: The output after passing through all the layers.
        """

        output = x

        for layer in self.layers:
            output = layer.forward(output)

        return output

    def backward(self, grad_output: Tensor) -> None:
        """
        Execute backpropagation through all avaible layers of the model.

        :param grad_output: The gradient of the loss function with
        respect to the model output
        """

        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)

    def get_parameters(self) -> List[Tensor]:
        """
        Return all model parameters (weights and biases).

        :return: List of tensors representing the parameters
        """
        parameters = []
        for layer in self.layers:
            parameters.extend(layer.get_parameters())

        return parameters

    def zero_grad(self) -> None:
        """
        Reset all model layer parameters to prepare for
        a new training interpolation.
        """

        for layer in self.layers:
            layer.zero_grad()

    def load(self, file_path: str) -> None:
        """
        Load the model's weights and biases from an HDF5 file.

        :param file_path: File path where is the target parameters.
        """

        with h5py.File(file_path, 'r') as file:
            for idx, layer in enumerate(self.layers):
                layer_group = file[f'layer_{idx}']
                parameters = layer.get_parameters()
                for i, param in enumerate(parameters):
                    param.data = layer_group[f'param_{i}'][:]

        print('Model loaded successfully')

    def save(self, file_path: str) -> None:
        """
        Save model's weights and biases in an HDF5 file.

        :param file_path: File path where is the target parameters.
        """

        with h5py.File(file_path, 'w') as file:
            for idx, layer in enumerate(self.layers):
                layer_group = file.create_group(f'layer_{idx}')
                parameters = layer.get_parameters()
                for i, param in enumerate(parameters):
                    param.data = layer_group[f'layer_{i}'][:]

        print(f'Model save in {file_path}')

    def train_on_batch(self, X: Tensor, y: Tensor) -> Dict[str, float]:
        """
        Train the model on a single batch.

        :param X: The input tensor.
        :param y: Tensor with labels.
        :return: Dict containing the loss and metrics computed for the batch.
        """

        # forward pass
        predictions = self.forward(X)

        # loss
        loss_result = self.loss.forward(predictions, y)

        # backpropagation pass
        grad_output = self.loss.backward()
        self.backward(grad_output)

        # parameters updated
        self.optimizer.step()
        self.optimizer.zero_grad()

        # metrics calculated
        metrics_results = {
            metric_name: metric_fn(y, predictions)
            for metric_name, metric_fn in self.metrics.items()
        }

        metrics_results['loss'] = loss_result
        return metrics_results

    def _train_epoch(
        self,
        X: Tensor,
        y: Tensor,
        num_samples: int,
        batch_size: int
    ) -> Tuple[float, Dict[str, float]]:
        """
        Train the model on all batches of an epoch
        """

        epoch_loss = 0
        epoch_metrics = {key: 0 for key in self.metrics.keys()}

        for i in range(0, num_samples, batch_size):
            X_batch = X[i:i + batch_size]
            y_batch = y[i:i + batch_size]

            # Train the model on a single batch
            batch_results = self.train_on_batch(X_batch, y_batch)

            # Accumulate the loss and metrics
            epoch_loss += batch_results['loss']
            for metric_name in epoch_metrics:
                epoch_metrics[metric_name] += batch_results[metric_name]

        # Metrics and loss avarage by epoch
        epoch_loss /= (num_samples // batch_size)
        for metric_name in epoch_metrics:
            epoch_metrics[metric_name] /= (num_samples // batch_size)

        return epoch_loss, epoch_metrics

    def _shuffle_data(self, X: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Suffle the data and labels together.
        """
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        return X[indices], y[indices]

    def _initialize_history(self) -> Dict[str, List[float]]:
        """
        Inicialize the history dict to store the losses and metrics.
        """
        history = {key: [] for key in self.metrics.keys()}
        history['loss'] = []
        return history

    def _update_history(
        self,
        history: Dict[str, List[float]],
        epoch_loss: float,
        epoch_metrics: Dict[str, float]
    ) -> None:
        """
        Update the history at the end of each epoch.
        """
        history['loss'].append(epoch_loss)
        for metric_name, metric_value in epoch_metrics.items():
            history[metric_name].append(metric_value)

    def fit(
        self,
        X: Tensor,
        y: Tensor,
        epochs: int,
        batch_size: int = 32,
        shuffle: bool = True,
        callbacks: Optional[List[Callable]] = None
    ) -> Dict[str, List[float]]:
        """
        Train the model over multiple batches for a number of epochs.

        :param X: The input tensor.
        :param y: Tensor with labels.
        :param epochs: Number of epochs.
        :param batch_size: The batch size.
        :param shuffle: If true, shuffle the data at each epoch.
        :param callbacks: List of callback functions to be executed each epoch.
        :result: History containing the loss and metrics by epoch
        """

        history = self._initialize_history()
        num_samples = X.shape[0]

        for epoch in range(epochs):
            if shuffle:
                X, y = self._shuffle_data(X, y)

            epoch_loss, epoch_metrics = self._train_epoch(
                X, y, num_samples, batch_size
            )
            self._update_history(history, epoch_loss, epoch_metrics)

            # Is there, execute callcack
            if callbacks:
                for callback in callbacks:
                    callback(epoch, history)

            print(f'Epoch {epoch + 1}/{epochs} |Loss: {epoch_loss}\
                  - {epoch_metrics}')

        return history

    def evaluate(self, X: Tensor, y: Tensor) -> None:
        """
        Evaluate the model on a datset.

        :param X: The input tensor.
        :param y: Tensor with labels.
        :return: Dict containing the calculated loss and metrics.
        """

        predictions = self.forward(X)
        loss_result = self.loss.forward(predictions, y)

        metrics_result = {
            metric_name: metric_fn(predictions, y)
            for metric_name, metric_fn in self.metrics.items()
        }

        metrics_result['loss'] = loss_result
        return metrics_result

    def predict(self, X: Tensor) -> Tensor:
        """
        Peform the prediction an a dataset.

        :param X: The input tensor.
        :return: Tensor with predictions.
        """
        return self.forward(X)
