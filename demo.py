from cuicui.neuralnet.model import Model
from cuicui.neuralnet.layers import Dense
from cuicui.neuralnet.activations import Sigmoid
from cuicui.neuralnet.optimizers import SGD
from cuicui.neuralnet.losses import MSE
from cuicui.neuralnet.core import Tensor

import numpy as np


def accuracy(y_true: Tensor, y_pred: Tensor) -> float:
    return float(np.mean(np.abs(y_true - y_pred) < 0.1))

# Exemplo de callback
def print_loss_callback(epoch, history):
    print(f"Callback: Epoch {epoch + 1}, Loss: {history['loss'][-1]}")

# Criar o modelo
model = Model(layers=[
    Dense(input_size=5, output_size=10, activation=Sigmoid()),
    Dense(input_size=10, output_size=1)
])

# Compilar o modelo
model.compile(
    loss=MSE(),
    optimizer=SGD(model.get_parameters(), learning_rate=0.01),
    metrics=[accuracy]
)

# Dados de entrada
X = Tensor(np.random.randn(100, 5), requires_grad=True)  # 100 amostras com 5 características
y = Tensor(np.random.randn(100, 1))  # 100 rótulos (valores reais)

# Treinar o modelo por 10 épocas com batch size de 32 e usando o callback
history = model.fit(X, y, epochs=10, batch_size=32, shuffle=False, callbacks=[print_loss_callback])