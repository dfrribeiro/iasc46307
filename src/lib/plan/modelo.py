class ModeloMundo:
    """
    Classe abstrata que define o modelo do mundo.
    O modelo do mundo é o ambiente da perspectiva do agente.
    """

    @property
    def S(self):
        """
        Retorna o conjunto de estados possíveis do mundo.
        """
        raise NotImplementedError

    @property
    def A(self):
        """
        Retorna o conjunto de ações possíveis do mundo.
        """
        raise NotImplementedError

    def T(self, estado, acao):
        """
        Retorna a transição de estado dado um estado e uma ação.

        Uma transição pode ser determinística ou estocástica:

        Uma transição estocástica é representada por uma distribuição de
        probabilidade sobre os estados possíveis, enquanto que uma transição
        determinística é representada apenas por um estado.

        Esta função retorna sempre apenas um estado, sendo que deve escolher
        aleatoriamente um estado de acordo com a distribuição de probabilidade
        para transições estocásticas.
        """
        raise NotImplementedError

    def distancia(self, estado1, estado2):
        """
        Retorna a distância entre dois estados.
        """
        raise NotImplementedError
