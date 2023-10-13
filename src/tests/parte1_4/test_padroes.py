import numpy as np
from keras.layers import Dense, Input
from lib.rede_neuronal_keras import RedeNeuronal
import matplotlib.pyplot as plt

dim_padrao = (4, 4)  # 16 bits / dimensões / vetores de caraterísticas
padrao_A = int("0000011001100000", 2)
padrao_B = int("0110100110010110", 2)

# Todas as combinações possíveis de 16 bits, organizadas em matrizes 4x4
combinacoes = np.array(
    [
        [int(digito) for digito in format(n, "016b")]
        for n in range(2 ** np.prod(dim_padrao))
    ]
).reshape(-1, 4, 4, 1)

# Visualização dos padrões
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Padrão A")
plt.imshow(combinacoes[padrao_A], cmap=plt.cm.gray)

plt.subplot(1, 2, 2)
plt.title("Padrão B")
plt.imshow(combinacoes[padrao_B], cmap=plt.cm.gray)

plt.show()


# Definição da arquitetura da rede
entrada = Input(shape=(16,), name="camada_entrada")
camada_escondida = Dense(
    5, activation="tanh"
)  # Pelo menos 3 neurónios (>= número de classes + 1)
# Como os pesos podem tomar valores negativos,
# escolheu-se a função de ativação tangente hiperbólica
saida = Dense(2, activation="sigmoid")
# Como os valores de saída são probabilidades, escolheu-se a função de ativação sigmoide

rede = RedeNeuronal()
rede.juntar(entrada)
rede.juntar(camada_escondida)
rede.juntar(saida)

# Dados de treino
X = np.array([combinacoes[padrao_A].flatten(), combinacoes[padrao_B].flatten()])
y = np.array([[1, 0], [0, 1]])

# Treino da rede
# A rede precisa de poucas épocas para aprender o problema graças à combinação de
# parâmetros de treino adequados e tamanho da camada escondida.
# Os hiperparâmetros foram escolhidos baseados na solução do problema XOR (parte 1.3).
# O problema não é o mesmo, mas é semelhante suficiente para justificar a escolha.

erros = rede.treinar(
    entradas=X,
    saidas=y,
    epocas=20,
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


def gerar_mutacao(padrao, dim, fator_exponencial=1.2):
    """
    Gera uma mutação do padrão fornecido.
    Começa por esoclher a quantidade de bits a alterar com uma distribuição exponencial
    (mais provável que se alterem poucos bits do que muitos, mínimo 1).
    Em seguida, escolhe aleatoriamente as posições dos bits a alterar.
    Por fim, altera os bits selecionados com a operação XOR bit-a-bit.

    Parâmetros:
        padrao: Padrão a ser mutado (número inteiro).
        dim: Dimensão do padrão (porque zeros à esquerda não são contados para inteiros)

    Retorna:
        Padrão mutado (número inteiro).
    """
    max_alterar = dim // 2
    probabilidades = [1 / (N + 1) ** fator_exponencial for N in range(max_alterar)]
    probabilidades /= np.sum(probabilidades)

    # Gerar um número aleatório de bits a alterar
    num_alterar = np.random.choice(range(max_alterar), p=probabilidades) + 1

    # Criar uma máscara com o número selecionado de bits a 1
    posicoes_alterar = np.random.choice(range(dim), size=num_alterar, replace=False)
    mascara = np.zeros(dim, dtype=int)
    mascara[posicoes_alterar] = 1

    mascara_bit = int("".join(map(str, mascara)), 2)

    # Aplicar a operação XOR para inverter os bits selecionados
    padrao_mutacao = padrao ^ mascara_bit

    return padrao_mutacao


# Dados de teste
num_instancias = 500
num_padrao_A = num_instancias // 2  # metade das instâncias são do padrão A
# caso este valor seja ímpar, o padrão B terá a que resta da divisão inteira

Xteste = np.empty((0, 16), dtype=int)

# Gerar mutações para o padrão A e acrescentar ao conjunto de teste
for _ in range(num_padrao_A):
    mutacaoA = combinacoes[gerar_mutacao(padrao_A, dim=16)].flatten()
    Xteste = np.vstack((Xteste, mutacaoA))

# Gerar mutações para o padrão B e acrescentar ao conjunto de teste
for _ in range(num_instancias - num_padrao_A):
    mutacaoB = combinacoes[gerar_mutacao(padrao_B, dim=16)].flatten()
    Xteste = np.vstack((Xteste, mutacaoB))

# As saídas esperadas do conjunto de teste
yteste = np.array([[1, 0]] * num_padrao_A + [[0, 1]] * (num_instancias - num_padrao_A))

# Previsão
yn_teste = rede.prever(Xteste)
desvio = np.abs(yn_teste - yteste)

# Resultados da previsão
[
    print(
        f"{vetor[0]} => {vetor[1]} {np.round(vetor[1])} => incerteza {np.sum(vetor[2])}"
    )
    for vetor in zip(Xteste, yn_teste, desvio)
]

# Matriz de confusão
yn_teste_arredondado = np.round(yn_teste)

true_labels = np.argmax(yteste, axis=1)
predicted_labels = np.argmax(yn_teste_arredondado, axis=1)

verdadeiro_A = np.sum((true_labels == 0) & (predicted_labels == 0))
falso_A = np.sum((true_labels == 0) & (predicted_labels == 1))
verdadeiro_B = np.sum((true_labels == 1) & (predicted_labels == 1))
falso_B = np.sum((true_labels == 1) & (predicted_labels == 0))

# Resultados
print(
    f"Padrão A: {verdadeiro_A} verdadeiros, {falso_A} falsos",
    f"Padrão B: {verdadeiro_B} verdadeiros, {falso_B} falsos",
    sep="\n",
)
