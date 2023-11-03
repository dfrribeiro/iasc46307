import itertools
import numpy as np
from keras.layers import Input, Dense
from lib.rna.rede_neuronal_keras import RedeNeuronal
import matplotlib.pyplot as plt

# O objetivo deste código é avaliar o efeito de diferentes hiperparâmetros no treino de
# uma rede neuronal perceptrão multicamada para resolver o problema XOR.

# Os hiperparâmetros a testar são:
# - Taxa de aprendizagem
#   - Valores: 0.001, 0.01, 0.1, 0.5, 1
#   A taxa de aprendizagem controla a rapidez com que os pesos são atualizados.
#   No contexto da descida de gradiente, a taxa de aprendizagem é o tamanho do passo
# que é dado na direção do gradiente.
#   Uma taxa de aprendizagem muito pequena pode levar a um treino muito lento, enquanto
# que uma taxa de aprendizagem muito grande pode levar a oscilações no erro que podem
# impedir a convergência.

# - Termo de momento
#   - Valores: 0, 0.5, 0.9, 0.99
#   O termo de momento controla a influência dos pesos anteriores na atualização dos
# pesos. Um termo de momento de 0 significa que apenas o gradiente atual é considerado
# na atualização dos pesos, enquanto que um termo de momento de 1 significa que apenas
# os pesos anteriores são considerados.
#   Um termo de momento muito pequeno pode levar a um treino muito lento, enquanto que
# um termo de momento muito grande pode levar a oscilações no erro que podem impedir a
# convergência.

# - Ordem de apresentação dos dados de treino
#   - Valores: original (`shuffle=False`), aleatória (`shuffle=True`)
#   A ordem de apresentação dos dados de treino pode ser original ou aleatória. No caso
# de ser aleatória, os dados de treino são apresentados à rede neuronal numa ordem
# aleatória a cada época. No caso de ser original, os dados de treino são apresentados
# à rede neuronal na mesma ordem a cada época.
#   A ordem de apresentação dos dados de treino pode ter um efeito no treino da rede
# neuronal. No caso de ser aleatória, a rede neuronal pode aprender mais rapidamente
# porque os dados de treino são apresentados de forma mais variada. No caso de ser
# original, a rede neuronal pode aprender mais lentamente porque os dados de treino são
# apresentados sempre na mesma ordem.

# O número de neurónios na camada escondida é fixo, e o número de saídas também se
# mantém. A complexidade da rede poderia ser aumentada, para obter melhores resultados.
# A procura no espaço de estados é muito exigente com 2 neurónios na camada escondida:
# muitos ótimos locais - com mais neurónios, seria mais fácil encontrar um ótimo global.
# Ao receber ruído, a rede perde a capacidade de generalização,
# e ocorre sobreparametrização.

# Note-se que 2 neurónios na camada escondida é o mínimo suficiente mas ineficiente
# para resolver o problema XOR. Aumentando o número de neurónios, a rede neuronal pode
# aprender mais rapidamente e com menos erros.
# Semelhantemente, a rede neuronal pode aprender mais rapidamente e com menos erros
# se o número de camadas escondidas for aumentado, sendo em geral mais eficiente do que
# aumentar o número de neurónios numa única camada escondida.


# Definição da arquitetura da rede (os pesos ainda não são inicializados a este ponto)
def criar_modelo():
    rede = RedeNeuronal()
    camada_entrada = Input(shape=(2,), name="camada_entrada")
    camada_escondida = Dense(2, activation="tanh", name="camada_escondida")
    camada_saida = Dense(1, activation="sigmoid", name="camada_saida")
    rede.juntar(camada_entrada)
    rede.juntar(camada_escondida)
    rede.juntar(camada_saida)
    return rede


def colecionar_erros(
    X,
    y,
    valores_taxa_aprend,
    valores_momento,
    valores_ordem,
    num_repeticoes,
    num_epocas,
):
    dim_matriz = (
        # *map(len, hiperparametros),
        len(valores_taxa_aprend),
        len(valores_momento),
        len(valores_ordem),
        num_repeticoes,
        num_epocas,
    )

    matriz = np.empty(dim_matriz, dtype=np.float64)
    matriz_param = itertools.product(
        valores_taxa_aprend, valores_momento, valores_ordem, range(num_repeticoes)
    )

    cabecalhos = ("Repetição", "Taxa Apr.", "Momento", "Ordem")
    linha_cabeca = " | ".join([f"{header:<9}" for header in cabecalhos])
    print(linha_cabeca)
    print("-" * len(linha_cabeca))

    # Para cada com combinação de hiperparâmetros,
    # treinar a rede e guardar os resultados
    for eta, alpha, chi, rep in matriz_param:
        rede = criar_modelo()

        table_row = (
            f"{rep+1:>4}/{num_repeticoes:<4}",
            f"{eta:<9}",
            f"{alpha:<9}",
            f"{('aleatória' if chi else 'original'):<9}",
        )
        row_string = " | ".join(table_row)
        print(row_string)

        matriz[
            valores_taxa_aprend.index(eta),
            valores_momento.index(alpha),
            valores_ordem.index(chi),
            rep,
        ] = rede.treinar(
            entradas=X,
            saidas=y,
            epocas=num_epocas,
            taxa_aprendizagem=eta,
            momento=alpha,
            ordem_aleatoria=chi,
        )

        # yn = rede.prever(X_treino)
        # print(yn)
    return matriz


def mostrar_efeito_param(
    erros, eixo, valores, nome, titulo, ficheiro, repr_erros_por_graf=20
):
    """
    Expõe o efeito de um parâmetro, definido pelo eixo e uma gama de valores,
    no treino da rede neuronal.

    Parâmetros:
        erros: array shape (num_epocas,)
    """
    num_epocas = erros.shape[-1]
    plt.figure(figsize=(12, 6))

    # para cada valor possível no parâmetro
    for idx, v in enumerate(valores):
        erros_param = np.take(erros, indices=idx, axis=eixo)
        dim_epoca = tuple(range(erros_param.ndim - 1))
        erro_medio = np.mean(erros_param, axis=dim_epoca)
        desvio_erro = np.std(erros_param, axis=dim_epoca)

        dom = np.arange(num_epocas)
        plt.errorbar(
            x=dom,
            y=erro_medio,
            yerr=desvio_erro,
            label=f"{v}",
            alpha=0.5,
            fmt=":",
            capsize=3,
            capthick=1,
            errorevery=max(1, num_epocas // repr_erros_por_graf),
        )
        plt.fill_between(
            x=dom,
            y1=erro_medio - desvio_erro,
            y2=erro_medio + desvio_erro,
            alpha=0.2,
        )

    plt.title(titulo)
    plt.xlabel("Época")
    plt.ylabel("Erro quadrático médio")
    plt.legend(title=nome)
    plt.savefig(ficheiro)
    plt.show()


if __name__ == "__main__":
    # Dados de treino XOR
    X_treino = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_treino = np.array([[0], [1], [1], [0]])

    # Hiperparâmetros de teste
    # Uma taxa de aprendizagem acima de 1 significa que os pesos são atualizados com
    # valores superiores aos valores atuais, o que pode levar a oscilações no erro.
    valores_taxa_aprend = list(reversed([0.001, 0.01, 0.1, 0.5, 1]))
    valores_momento = list(reversed([0, 0.5, 0.9, 0.99]))
    valores_ordem = [False, True]

    # Matriz para guardar os resultados (taxa_aprend, momento, ordem, repetição, época)
    matriz = colecionar_erros(
        X_treino,
        y_treino,
        valores_taxa_aprend,
        valores_momento,
        valores_ordem,
        num_repeticoes=10,
        num_epocas=5000,
    )

    # ultima época (taxa, momento, ordem, repetição)
    erros_final = np.take(matriz, indices=-1, axis=-1)

    # Taxa de aprendizagem
    mostrar_efeito_param(
        matriz,
        0,
        valores_taxa_aprend,
        "Taxa de aprendizagem",
        "Evolução de perda para cada taxa de aprendizagem",
        "out/efeito_taxa_aprend.png",
    )

    # Termo de momento
    mostrar_efeito_param(
        matriz,
        1,
        valores_momento,
        "Termo de momento",
        "Evolução de perda para cada termo de momento",
        "out/efeito_momento.png",
    )

    # Ordem de apresentação
    mostrar_efeito_param(
        matriz,
        2,
        valores_ordem,
        "Aleatoriedade",
        "Evolução de perda para apresentação aleatória ou não",
        "out/efeito_aleatorio.png",
    )

    # Análise de resultados
    mediana_taxa = np.median(erros_final, axis=(1, 2, 3))
    mediana_momento = np.median(erros_final, axis=(0, 2, 3))
    mediana_ordem = np.median(erros_final, axis=(0, 1, 3))
    mediana_combinacao = np.median(erros_final, axis=(3,))

    idx_taxa = np.argmin(mediana_taxa)
    idx_momento = np.argmin(mediana_momento)
    idx_ordem = np.argmin(mediana_ordem)
    idx_combinacao = np.unravel_index(
        np.argmin(mediana_combinacao), mediana_combinacao.shape
    )

    melhor_taxa = valores_taxa_aprend[idx_taxa]
    melhor_momento = valores_momento[idx_momento]
    melhor_ordem = valores_ordem[idx_ordem]
    melhor_combinacao = (
        valores_taxa_aprend[idx_combinacao[0]],
        valores_momento[idx_combinacao[1]],
        valores_ordem[idx_combinacao[2]],
    )

    erro_medio_combinacao = np.mean(
        erros_final[idx_combinacao[0], idx_combinacao[1], idx_combinacao[2], :]
    )
    erro_medio_taxa = np.mean(np.take(erros_final, indices=idx_taxa, axis=0))
    erro_medio_momento = np.mean(np.take(erros_final, indices=idx_momento, axis=1))
    erro_medio_ordem = np.mean(np.take(erros_final, indices=idx_ordem, axis=2))

    print(
        f"""
        Melhor = Menor mediana de erros finais
        Melhor taxa de aprendizagem: {melhor_taxa} (Erro médio: {erro_medio_taxa})
        Melhor momento: {melhor_momento} (Erro médio: {erro_medio_momento})
        Melhor ordem: {melhor_ordem} (Erro médio: {erro_medio_ordem})

        Melhor combinação: {melhor_combinacao}
        (Erro médio: {erro_medio_combinacao})
        """
    )

""" Resultado no. 1
Repetição | Taxa Apr. | Momento   | Ordem    
---------------------------------------------
   1/10   | 0.001     | 0         | original 
    ...
  10/10   | 1         | 0.99      | aleatória

        Melhor = Menor mediana de erros finais
        Melhor taxa de aprendizagem: 1 (Erro médio: 0.04746633621960152)
        Melhor momento: 0.99 (Erro médio: 0.05229693713951997)
        Melhor ordem: True (Erro médio: 0.09101280354760209)

        Melhor combinação: (0.5, 0.99, False)
        (Erro médio: 0.0416666782369918)
"""

"""Resultado no. 2
Repetição | Taxa Apr. | Momento   | Ordem    
---------------------------------------------
   1/10   | 1         | 0.99      | original 
    ...
  10/10   | 0.001     | 0         | aleatória

        Melhor = Menor mediana de erros finais
        Melhor taxa de aprendizagem: 1 (Erro médio: 0.0495478496972976)
        Melhor momento: 0.99 (Erro médio: 0.06559981622563255)
        Melhor ordem: True (Erro médio: 0.09165950857052912)

        Melhor combinação: (0.5, 0.99, True)
        (Erro médio: 0.03750001044935048)
"""

# TODO: Explorar custo temporal
