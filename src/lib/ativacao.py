import numpy as np


class FuncaoAtivacao:
    """
    Classe para representar uma função de ativação.
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

    Parâmetros:
        limiar: Limiar da função degrau.
    """

    def __init__(self, limiar):
        self.__limiar = limiar
        super().__init__()

    def aplicar(self, x):
        """
        Aplica a função degrau a uma matriz de valores x.

        Parâmetros:
            x: Valores a aplicar a função degrau.

        Retorna:
            Valores resultantes da aplicação da função degrau.
        """

        return np.heaviside(x, self.__limiar)

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Define o intervalo de valores
    x = np.linspace(-5, 5, 1000)

    # Calcula o valor da função degrau para cada valor do intervalo
    y = np.heaviside(x, 0)

    # Desenha o gráfico
    plt.plot(x, y, label='np.heaviside(x, 0)')
    plt.grid(True)
    plt.legend()
    plt.show()