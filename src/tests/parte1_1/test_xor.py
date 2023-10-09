import numpy as np


# Estas linhas definem a arquitetura do perceptrão multi-camada (MLP).
#
# * n_inputs: número de neurónios na camada de entrada
# * hidden_shape: um tuplo com o número de neurónios em cada camada escondida
# - o número de camadas escondidas é dado pelo comprimento do tuplo.
# - o número mínimo de camadas escondidas é 1 (uma camada escondida)
# - o número mínimo de neurónios em cada camada escondida é 1,
# mas deve ser maior ou igual ao o número de neurónios na camada de entrada
# * n_outputs: número de neurónios na camada de saída

n_inputs = 2
hidden_shape = (2,)
n_outputs = 1

# Estas linhas definem os dados de entrada e saída do perceptrão.
#
# * inputs: uma matriz com os dados de entrada
# - cada linha representa um exemplo
# - cada coluna representa uma dimensão ou atributo
# * labels: uma matriz com os dados de saída
# - cada linha representa um exemplo
# - cada coluna representa a saída de cada neurónio na camada de saída
# neste caso, a codificação é binária, ou seja, os valores são 0 ou 1
# a classificação é binária, ou seja, há apenas um neurónio na camada de saída
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # shape: (N, input_shape)
labels = np.array([[0], [1], [1], [0]])  # shape: (N, output_shape)

# Valores de inicialização dos pesos e pendores
# na camada escondida.
#
# * weights: uma matriz com os pesos das ligações entre neurónios
# - cada linha representa um neurónio na camada de entrada
# - cada coluna representa um neurónio na camada escondida
# * biases: um vetor com os pendores dos neurónios
# - cada elemento representa um neurónio na camada escondida
# Num caso habitual os valores costumam ser inicializados aleatoriamente,
# baseados numa distribuição.
# Neste caso, os pesos são definidos com base nas fronteiras de decisão
# ótimas para resolver o problema XOR.
# Definem os parâmetros de duas (uma para cada entrada) retas para cada neurónio na
# camada escondida:
# Os pesos são o declive das retas, e os pendores são o ponto de interceção
weights = np.array([[1, -1], [-1, 1]])  # shape: (input_shape, hidden_shape)
biases = np.array([-0.5, -0.5])  # shape: (hidden_shape, )

# Valores de inicialização dos pesos e pendores
# na camada de saída.
#
# * output_weights: uma matriz com os pesos das ligações entre neurónios
# - cada linha representa um neurónio na camada escondida
# - cada coluna representa um neurónio na camada de saída
# * output_bias: um vetor com os pendores dos neurónios
# - cada elemento representa um neurónio na camada de saída
# Para o neurónio de saída, duas retas são combinadas com as saídas da camada escondida.
output_weights = np.array([[1], [1]])  # shape: (hidden_shape, output_shape)
output_bias = -0.5  # shape: (output_shape, )


# Esta função define uma função de ativação.
# neste caso, a função de ativação é uma função degrau.
# a função degrau é uma função que retorna 0 se o valor de entrada for negativo
# e 1 se o valor de entrada for positivo.
def activation_degree(x):
    return np.heaviside(x, 0)


# Esta função define o processo de ativação de um neurónio.
# a ativação de um neurónio é o resultado da aplicação da função de ativação
# ao valor de saída do neurónio (y), que é calculado como o produto interno
# entre as entradas e pesos somado com os pendores.
def activate(inputs, weights, biases, activation_function):
    y = np.dot(inputs, weights) + biases
    return activation_function(y)


# Para cada entrada, calcula a saída do perceptrão, propagando de camada em camada.
outputs = np.ones((inputs.shape[0], 1))
for i, x in enumerate(inputs):
    hidden = activate(x, weights, biases, activation_degree)
    output = activate(hidden, output_weights, output_bias, activation_degree)
    outputs[i] = output

# Imprime os resultados para cada entrada.
for x, y in zip(inputs, outputs):
    print(x, "=>", y)
