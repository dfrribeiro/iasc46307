class MetodoAprendizagem:
    def __init__(self):
        raise NotImplementedError


class QLearning(MetodoAprendizagem):
    """
    O algoritmo Q-Learning é um algoritmo de aprendizagem por reforço off-policy.
    Isto significa que o agente aprende a política ótima enquanto seleciona a sua ação
    com uma política diferente.
    Off-policy: garante a otimização no infinito ao contrário do sarsa
    Agir no mundo, epsilon greedy, para aprender, greedy
    Aprende-se da a' ação que maximiza o Q estimação de valor para s_
    """

    def __init__(self, memoria, estrategia, alpha, gamma):
        super().__init__()
        self.__memoria = memoria
        self.__estrategia = estrategia
        self._alpha = alpha
        self._gamma = gamma

    def aprender(self, estado, acao, reforco, estado_seguinte, acao_seguinte=None):
        """
        Propaga o reforço para a ação tomada no estado dado.
        """

        acao_seguinte = self.__estrategia.acao_sofrega(estado_seguinte)
        qsa = self.__memoria[estado, acao]  # Estimativa atual
        qsnan = self.__memoria[estado_seguinte, acao_seguinte]

        delta = reforco + self._gamma * qsnan - qsa  # Erro de estimativa
        q = qsa + self._alpha * delta  # Nova estimativa
        self.__memoria.atualizar(estado, acao, q)
