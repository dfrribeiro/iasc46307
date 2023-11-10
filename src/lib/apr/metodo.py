class MetodoAprendizagem:  # chamada AprendRef na arquitetura
    def __init__(self, memoria, sel_acao, alpha, gamma):
        # Injeção de dependências
        self._memoria = memoria
        self._sel_acao = sel_acao

        self._alpha = alpha
        self._gamma = gamma


class QLearning(MetodoAprendizagem):
    """
    O algoritmo Q-Learning é um algoritmo de aprendizagem por reforço off-policy.
    Isto significa que o agente aprende a política ótima enquanto seleciona a sua ação
    com uma política diferente.

    Off-policy garante a otimização no infinito, ao contrário do SARSA.
    Ou seja, para agir no mundo utiliza Epsilon-Geedy, mas para aprender, Greedy.

    Aprende-se da `a'` (ação seguinte)
    que maximiza o `Q` (estimação de valor)
    para `s'` (estado seguinte).
    """

    def aprender(self, estado, acao, reforco, estado_seguinte, acao_seguinte=None):
        """
        Propaga o reforço para a ação tomada no estado dado.
        """

        acao_seguinte = self._sel_acao.acao_sofrega(estado_seguinte)
        qsa = self._memoria[estado, acao]  # Estimativa atual
        qsnan = self._memoria[estado_seguinte, acao_seguinte]

        delta = reforco + self._gamma * qsnan - qsa  # Erro de estimativa
        q = qsa + self._alpha * delta  # Nova estimativa
        self._memoria.atualizar(estado, acao, q)
