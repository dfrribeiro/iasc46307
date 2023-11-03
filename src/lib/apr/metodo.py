class MetodoAprendizagem:
    def __init__(self):
        raise NotImplementedError


class QLearning(MetodoAprendizagem):
    """
    O algoritmo Q-Learning é um algoritmo de aprendizagem por reforço off-policy.
    Isto significa que o agente aprende a política ótima enquanto seleciona a sua ação
    com uma política diferente.
    """

    def __init__(self, memoria, estrategia, alpha, gamma):
        super().__init__()
        self.__memoria = memoria
        self.__estrategia = estrategia
        self._alpha = alpha
        self._gamma = gamma

    def aprender(self, estado, acao, reforco, estado_seguinte=None):
        # off-policy: garante a otimização no infinito ao contrário do sarsa
        # agir no mundo, epsilon greedy, para aprender, greedy
        # aprende-se da a' ação que maximiza o Q estimação de valor para s_
        raise NotImplementedError
