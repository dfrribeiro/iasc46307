from collections import namedtuple
from lib.plan.modelo import ModeloMundo

Estado = namedtuple("Estado", ["x", "y"])
Acao = namedtuple("Acao", ["dx", "dy"])


class ModeloMundo2D(ModeloMundo):
    """
    Especialização do modelo do mundo para ambientes 2D.
    O modelo do mundo é representado por um conjunto de estados e um conjunto
    de ações possíveis. Os estados são representados por coordenadas (x, y) e
    as ações por vetores (dx, dy).
    """

    def __init__(self):
        self.__estados = set()
        self.__acoes = [Acao(0, 1), Acao(0, -1), Acao(1, 0), Acao(-1, 0)]
        # self.__xmax, self.__ymax = None, None

    @property
    def S(self):
        return self.__estados

    @property
    def A(self):
        return self.__acoes

    def T(self, estado, acao):
        """
        Retorna a transição de estado dado um estado e uma ação,
        None se existir uma colisão.
        """
        prox_estado = Estado(estado.x + acao.dx, estado.y + acao.dy)
        return prox_estado if self.__estado_valido(prox_estado) else None

    def distancia(self, estado1, estado2):
        """
        Retorna a distância de Manhattan entre dois estados.
        O uso da distância de Manhattan justifica-se com as ações do agente
        serem limitadas a movimentos ortogonais (norte, sul, este, oeste).
        """
        return abs(estado1.x - estado2.x) + abs(estado1.y - estado2.y)

    @property
    def x_max(self):
        return max([s.x for s in self.__estados])

    @property
    def y_max(self):
        return max([s.y for s in self.__estados])

    def atualizar(self, percepcao):
        """
        Atualiza o modelo do mundo com base na percepção do agente.
        """
        pass

    def obter_posicoes_alvo(self):
        """
        Retorna uma lista com as posições dos alvos.
        """
        pass

    def __simular_acao(self, estado, acao):
        """
        Simula a ação no estado dado e retorna o estado resultante.
        """
        pass

    def __estado_valido(self, estado):
        """
        Retorna True se o estado dado é válido (não é uma colisão e está
        dentro dos limites do mundo).
        """
        return estado in self.__estados
