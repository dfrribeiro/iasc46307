import numpy as np
from test_xor import xor

melhor_taxa = 0.1  # do efeito_1
melhor_momento = 0.5  # do efeito_2
melhor_aleatorio = None
menor_erro = np.inf
repeticoes = 5

for a in [False, True]:
    erros = np.ones(repeticoes) * np.inf

    for i in range(repeticoes):
        erros[i] = xor(eta=melhor_taxa, alpha=melhor_momento)
    media_erro = np.sum(erros) / repeticoes

    if media_erro < menor_erro:
        menor_erro = media_erro
        melhor_aleatorio = a

print(f"Melhor aleatorio: {melhor_aleatorio} com média de erro {menor_erro}")
