import numpy as np
from test_xor import xor

momentos_ = [0, 0.1, 0.5, 0.9, 0.99]
melhor_taxa = 0.1  # do efeito_1
melhor_momento = None
menor_erro = np.inf
repeticoes = 5

for m in momentos_:
    erros = np.ones(repeticoes) * np.inf

    for i in range(repeticoes):
        erros[i] = xor(eta=melhor_taxa, alpha=m)
    media_erro = np.sum(erros) / repeticoes

    if media_erro < menor_erro:
        menor_erro = media_erro
        melhor_momento = m

print(f"Melhor momento: {melhor_momento} com média de erro {menor_erro}")
