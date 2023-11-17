import random
from collections import namedtuple
from lib.apr.metodo import QLearning


class QME(QLearning):
    """
    Método de Aprendizagem por Reforço com Memória de Experiências (QME).

    O algoritmo QME é um algoritmo de aprendizagem por reforço off-policy.
    Acrescenta ao Q-Learning uma memória de experiências que permite ao agente
    aprender com experiências passadas.
    """

    def __init__(self, memoria, sel_acao, alpha, gamma, num_sim, dim_max):
        super().__init__(memoria, sel_acao, alpha, gamma)
        self.__num_sim = num_sim
        self.__memoria_experiencia = MemoriaExperiencia(dim_max)

    def aprender(self, estado, acao, reforco, estado_seguinte, acao_seguinte=None):
        """
        Propaga o reforço para a ação tomada no estado dado.

        É feita uma amostra aleatória de experiências da memória de experiências
        e o algoritmo Q-Learning é aplicado a cada uma dessas experiências.

        O algoritmo Q-Learning é também aplicado à experiência atual, que é
        adicionada à memória de experiências antes de ser utilizada para treino.

        Por cada ação do agente no mundo, o agente realiza `n` simulações de
        aprendizagem Q-Learning com experiências passadas.
        """

        super().aprender(estado, acao, reforco, estado_seguinte)
        self.__memoria_experiencia.atualizar(
            Experiencia(estado, acao, reforco, estado_seguinte)
        )

        self.simular()

    def simular(self):
        """
        Simula `n` experiências e aplica um passo de aprendizagem Q-Learning
        a cada uma delas.
        """

        amostras = self.__memoria_experiencia.amostrar(self.__num_sim)
        for experiencia in amostras:
            super().aprender(
                experiencia.estado,
                experiencia.acao,
                experiencia.recompensa,
                experiencia.estado_seguinte,
            )


"""
Uma experiência é um conjunto de quatro elementos:
- estado
- ação
- reforço
- estado seguinte
As experiências são utilizadas para treinar o agente num contexto de simulação,
onde o agente não tem acesso ao ambiente e, portanto, não pode utilizar o
reforço para aprender. Neste contexto, o agente aprende com as experiências
que vai acumulando ao longo do tempo.
"""
Experiencia = namedtuple(
    "Experiencia", ["estado", "acao", "recompensa", "estado_seguinte"]
)


class MemoriaExperiencia:
    """
    A memória de experiências é uma lista ordenada de experiências.
    A memória é limitada a um número máximo de experiências.
    """

    def __init__(self, dim_max):
        self.__dim_max = dim_max
        self.__memoria = []

    def atualizar(self, experiencia):
        """
        Por critério de antiguidade: quando a memória atinge esse limite,
        a experiência mais antiga é descartada para dar lugar à mais recente.
        """

        if len(self.__memoria) >= self.__dim_max:
            self.__memoria.pop(0)
        self.__memoria.append(experiencia)

    def amostrar(self, tamanho):
        """
        Retorna uma amostra aleatória de experiências. O tamanho da amostra é
        limitado ao tamanho da memória. Se o tamanho da amostra for maior que o
        tamanho da memória, a função retorna uma amostra com todas as experiências
        da memória.
        A aleatoriedade é importante para evitar que o agente aprenda apenas com
        experiências recentes.
        A amostragem aleatória é feita sem reposição
        (não há repetição de experiências dentro da amostra).
        """
        n = min(tamanho, len(self.__memoria))
        return random.sample(self.__memoria, n)
