# O número de neurónios na camada escondida é fixo, e o número de saídas também se
# mantém. A complexidade da rede poderia ser aumentada, para obter melhores resultados.
# A procura no espaço de estados é muito exigente com 2 neurónios na camada escondida:
# muitos ótimos locais - com mais neurónios, seria mais fácil encontrar um ótimo global.
# Ao receber ruído, a rede perde a capacidade de generalização,
# e ocorre sobreparametrização.

import numpy as np
from test_xor import xor

taxas_ = [0.001, 0.01, 0.1, 0.5, 1]
melhor_taxa = None
menor_erro = np.inf
repeticoes = 5

for t in taxas_:
    erros = np.ones(repeticoes) * np.inf

    for i in range(repeticoes):
        erros[i] = xor(eta=t)
    media_erro = np.sum(erros) / repeticoes

    if media_erro < menor_erro:
        menor_erro = media_erro
        melhor_taxa = t

print(f"Melhor taxa: {melhor_taxa} com média de erro {menor_erro}")
