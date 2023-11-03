import numpy as np
from keras.layers import Dense, Input
from lib.rna.rede_neuronal_keras import RedeNeuronal
import matplotlib.pyplot as plt

print("-- PROBLEMA XOR -- (cod. binária)")


"""
Testa a rede XOR baseada nestes parâmetros, para perceber os seus efeitos.
"""
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# As saídas são binárias, logo a função de ativação sigmóide é a mais adequada.
# A função de ativação degrau não é diferenciável, logo não pode ser utilizada
# na retropropagação do erro (só é aplicável no "feedforward").
# Na camada escondida, como o erro pode ser negativo,
# a tangente hiperbólica ou ReLU são as mais adequadas.
camada_entrada = Input(shape=(2,), name="camada_entrada")
camada_escondida = Dense(2, activation="tanh", name="camada_escondida")
camada_saida = Dense(1, activation="sigmoid", name="camada_saida")

# Arquitetura da rede
rede = RedeNeuronal()
rede.juntar(camada_entrada)
rede.juntar(camada_escondida)
rede.juntar(camada_saida)

# Treino da rede
erros = rede.treinar(
    entradas=X,
    saidas=y,
    epocas=1000,
    taxa_aprendizagem=0.5,
    momento=0.99,
    ordem_aleatoria=False,
)
rede.mostrar()

# Previsão
yn = rede.prever(X)

# Resultados da previsão
[print(f"{vetor[0]} => {vetor[1]} {np.round(vetor[1])}") for vetor in zip(X, yn)]

# Gráfico de desempenho
plt.plot(erros)
plt.xlabel("Época")
plt.ylabel("Erro")
plt.show()
