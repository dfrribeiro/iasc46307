import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
import numpy as np

# Dados de entrada
x = np.array([[0,0],[0,1],[1,0],[1,1]])

# Função de ativação degrau
def custom_activation(x):
    return tf.cast(tf.math.greater(x, tf.constant([0.])), tf.float32)

# Arquitetura da rede
model = Sequential()

# função degrau não é diferenciável!
model.add(Dense(2, input_dim=2,
                kernel_initializer="zeros", activation=custom_activation,
                name='hidden_layer'))
model.add(Dense(1,
                kernel_initializer="zeros", activation=custom_activation,
                name='output_layer'))
# print(model.get_weights())

# Pesos e pendores
wb = [np.array([[1., -1.], [-1., 1.]]),
      np.array([-0.5, -0.5]),
      np.array([[1.], [1.]]),
      np.array([-0.5])]
model.set_weights(wb)
# print(model.get_weights())
model.save('models/xor.keras')

# Predição
y = model.predict(x)

for xi, yi in zip(x, y):
    print(xi, "=>", yi)


import matplotlib.pyplot as plt

# Gerar uma grelha de pontos
xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 100), np.linspace(-1.5, 1.5, 100))
grid_points = np.c_[xx.ravel(), yy.ravel()]

# Utilizar a rede neuronal para prever os valores da grelha
predictions = model.predict(grid_points)

# Transformar os valores da grelha numa matriz
predictions = predictions.reshape(xx.shape)

# Desenhar a fronteira de decisão
plt.contourf(xx, yy, predictions, levels=[-0.1, 0.1], alpha=0.3)

# Desenhar os dados de entrada
plt.scatter(x[:, 0], x[:, 1], c='black', marker='o', s=100, label="Dados de entrada")

plt.xlabel("Dimensão 1")
plt.ylabel("Dimensão 2")
plt.title("Fronteira de decisão (XOR binário)")
plt.grid()
plt.legend()
plt.show()