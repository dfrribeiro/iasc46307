import numpy as np


class CamadaDensa:
    """
    Representa uma camada densa de uma rede neuronal.

    Uma camada densa é definida por um conjunto de neurónios em que todos têm
    ligações de entrada para todos os neurónios da camada anterior e
    ligações de saída para todos os neurónios da camada seguinte.

    Parâmetros:
        dim_entrada: Dimensão de entrada da camada - quantidade de ligações que cada
        neurónio vai ter para a camada anterior
        dim_saida: Dimensão de saída da camada - quantidade de neurónios na camada
        funcao_ativacao: Função de ativação da camada.
        pesos: Pesos da camada, inicializados com uma distribuição normal.
        pendores: Pendores da camada, inicializados com uma distribuição normal.
    """

    def __init__(
        self,
        dim_entrada,
        dim_saida,
        funcao_ativacao=None,
    ):
        self.__funcao_ativacao = funcao_ativacao
        self.dim_entrada = dim_entrada
        self.dim_saida = dim_saida
        self.__pesos = np.random.randn(dim_entrada, dim_saida)
        self.__pendores = np.random.randn(dim_saida)

    @property
    def pesos(self):
        return self.__pesos

    @property
    def pendores(self):
        return self.__pendores

    def atualizar_pesos(self, pesos):
        """
        Atualiza os pesos da camada.

        Parâmetros:
            pesos: Novos pesos da camada.

        Exceções:
            AssertionError: Se a dimensão dos pesos não for igual à dimensão dos pesos
            da camada.
        """

        assert pesos.shape == self.__pesos.shape
        self.__pesos = pesos

    def atualizar_pendores(self, pendores):
        """
        Atualiza os pendores da camada.

        Parâmetros:
            pendores: Novos pendores da camada.

        Exceções:
            AssertionError: Se a dimensão dos pendores não for igual à dimensão dos
            pendores da camada.
        """

        assert pendores.shape == self.__pendores.shape
        self.__pendores = pendores

    def ativar(self, entradas):
        """
        Aplica a função de ativação, pesos e pendores da camada às entradas fornecidas.

        Parâmetros:
            entradas: Dados de entrada na camada.

        Retorna:
            Saídas da camada depois de ativada.

        """

        if self.__funcao_ativacao is None:
            return entradas

        y = np.dot(entradas, self.__pesos) + self.__pendores
        return self.__funcao_ativacao.aplicar(y)

    def __str__(self):
        return f"""CamadaDensa(
            dim_entrada={self.dim_entrada},
            dim_saida={self.dim_saida},
            funcao_ativacao={self.__funcao_ativacao},
            pesos={self.pesos},
            pendores={self.pendores})"""


if __name__ == "__main__":
    camada = CamadaDensa(2, 2)
    print(camada)
