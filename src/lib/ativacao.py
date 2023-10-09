import numpy as np


class FuncaoAtivacao:
    """
    Classe para representar uma função de ativação.

    A função de ativação, associada a uma camada, é uma operação aplicada
    ao valor resultante da combinação linear dos pesos e pendores da camada,
    que é um vetor de valores reais com o mesmo tamanho da dimensão de saída
    da camada.

    A função de ativação tem o objetivo de introduzir não linearidade na rede.
    Para ser utilizada numa descida de gradiente, a função de ativação deve ser
    diferenciável.
    """

    def aplicar(self, x):
        """
        Aplica a função de ativação a uma matriz de valores x.

        Parâmetros:
            x: Valores a aplicar a função de ativação.
        """
        raise NotImplementedError


class Degrau(FuncaoAtivacao):
    """
    Classe para representar uma função de ativação degrau.

    A função degrau não é diferenciável, logo o uso na descida de gradiente,
    em específico na minimização do erro da retropropagação, é indefinida em 0.
    No entanto, a função degrau pode ser utilizada em redes neurais, em particular
    em perceptrões, para classificação binária.

    Parâmetros:
        limiar: O limiar da função degrau. Todos os valores de entrada x inferiores ao
        limiar serão mapeados para 0, e todos os valores de entrada superiores ou iguais
        ao limiar serão mapeados para 1.
    """

    def __init__(self, limiar):
        self.__limiar = limiar
        super().__init__()

    def aplicar(self, x):
        """
        Aplica a função degrau a uma matriz de valores x.

        Parâmetros:
            x: Valores a aplicar à função degrau.

        Retorna:
            Valores resultantes da aplicação da função degrau.
        """

        return np.heaviside(x, self.__limiar)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Define o intervalo de valores
    x = np.linspace(-5, 5, 1000)

    # Calcula o valor da função degrau para cada valor do intervalo
    y = np.heaviside(x, 0)

    # Desenha o gráfico
    # O eixo x é um intervalo de valores possíveis (não exclusivamente)
    # para os valores provenientes da combinação de pesos e pendores da camada
    # O eixo y é o valor resultante da função degrau
    plt.plot(x, y, label="np.heaviside(x, 0)")
    plt.grid(True)
    plt.legend()
    plt.show()
