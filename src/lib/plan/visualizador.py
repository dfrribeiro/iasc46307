import matplotlib.pyplot as plt
from lib.plan.modelo2d import Acao


class VisValorPol:
    """
    Classe que implementa um visualizador de valor e política.
    """

    def mostrar(self, x_max, y_max, V, politica):
        """
        Mostra os valores da política.
        """
        fig, grafico = plt.subplots()
        fig.suptitle("Valor e política")

        # Define eixo X
        X = range(x_max)

        # Define eixo Y
        Y = range(y_max)

        # Eixo Z (valor): lista de listas em que cada valor
        # corresponde ao valor do estado (x, y)
        Z = [[V.get((x, y), 0) for x in X] for y in Y]

        # Onde não há ação (ex. obstáculos), colocar Acao(0, 0)
        ACAO_POR_OMISSAO = Acao(0, 0)

        # Horizontal das setas
        DX = [[politica.get((x, y), ACAO_POR_OMISSAO).dx for x in X] for y in Y]

        # Vertical das setas
        DY = [[-politica.get((x, y), ACAO_POR_OMISSAO).dy for x in X] for y in Y]
        # No Matplotlib, o eixo Y cresce de baixo para cima, enquanto que no ambiente
        # do agente, o eixo Y cresce de cima para baixo. Assim, é necessário inverter
        # o eixo Y para que as imagens sejam consistentes.

        # Gera o gradiente de cores
        grafico.imshow(Z)

        # Política (setas)
        grafico.quiver(X, Y, DX, DY, scale_units="xy", scale=2)

        plt.show()
