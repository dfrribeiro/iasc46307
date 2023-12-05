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

    __ALVO = "+"
    __OBSTACULO = "#"
    # As constantes de elementos no ambiente são redefinidos aqui porque não existe
    # dependência entre modelo do mundo e a "realidade" do ambiente.

    def __init__(self):
        self.__estados = None
        self.__acoes = [Acao(0, 1), Acao(0, -1), Acao(1, 0), Acao(-1, 0)]
        self.__xmax, self.__ymax = None, None

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
        return self.__simular_acao(estado, acao)

    @property
    def xmax(self):
        """
        "Largura" do ambiente: é o valor máximo exclusive
        """
        return self.__xmax

    @property
    def ymax(self):
        """
        "Altura" do ambiente: é o valor máximo exclusive
        """
        return self.__ymax

    def distancia(self, estado1, estado2):
        """
        Retorna a distância de Manhattan entre dois estados.
        O uso da distância de Manhattan justifica-se com as ações do agente
        serem limitadas a movimentos ortogonais (norte, sul, este, oeste).
        """
        return abs(estado1.x - estado2.x) + abs(estado1.y - estado2.y)

    def atualizar(self, percepcao):
        """
        Atualiza o modelo do mundo com base na percepção do agente.

        A percepção do agente é uma lista de listas de carateres, em que cada carater
        representa um elemento do ambiente (vazio, agente, obstáculo, alvo).

        O modelo é atualizado dinamicamente, mas inclui restrições impostas pelo
        especialista. O especialista sabe que o agente não se pode movimentar sobre
        obstáculos.
        """
        self.__elementos = percepcao
        self.__ymax = len(percepcao)
        self.__xmax = len(percepcao[0])

        # Garantir que todas as linhas têm o mesmo tamanho
        assert all(len(linha) == self.x_max for linha in percepcao)

        # Alternativamente:
        # self.__xmax = max(len(linha) for linha in percepcao)

        # Filtra os estados válidos de entre de todas as posições do ambiente
        self.__estados = [
            Estado(x, y)
            for y in range(self.y_max)
            for x in range(self.x_max)
            if percepcao[y][x] != self.__OBSTACULO
        ]

    def obter_posicoes_alvo(self):
        """
        Retorna uma lista com as posições (x, y) dos alvos.
        """
        return [
            (x, y) for (x, y) in self.__estados if self.__elementos[y][x] == self.__ALVO
        ]

    def __simular_acao(self, estado, acao):
        """
        Simula a ação no estado dado e retorna o estado resultante.
        Gera um estado sucessor e retorna-o se o estado for válido (não é uma
        colisão e está dentro dos limites do mundo).
        """
        prox_estado = Estado(estado.x + acao.dx, estado.y + acao.dy)
        return prox_estado if self.__estado_valido(prox_estado) else None
        # É eficiente retornar None em vez do estado anterior, porque assim o algoritmo
        # de frente de onda não vai expandir estados repetidos.

    def __estado_dentro_limites(self, estado):
        """
        Retorna True se o estado dado está dentro dos limites do mundo.
        """
        return 0 <= estado.x < self.x_max and 0 <= estado.y < self.y_max

    def __estado_valido(self, estado):
        """
        Retorna True se o estado dado é válido (não é uma colisão e está
        dentro dos limites do mundo).
        O estado é valido se pertence à lista de estados válidos: a filtragem
        foi feita anteriormente na função atualizar.
        """
        return estado in self.__estados
