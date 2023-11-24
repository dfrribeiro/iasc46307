class Aprendizagem:
    """
    Interface para os métodos de aprendizagem.
    """

    def __init__(self):
        raise NotImplementedError

    def aprender(self, estado, acao, reforco, estado_seguinte, acao_seguinte=None):
        raise NotImplementedError

    def selecionar_acao(self, estado):
        raise NotImplementedError


class MecanismoAprendizagem(Aprendizagem):
    """
    Implementa os algoritmos de aprendizagem. Específico a Q-Learning com E-Greedy.

    O mecanismo utiliza uma memória de aprendizagem para guardar o conhecimento
    adquirido, sob a forma de pares estado-ação e valores de reforço.

    As ações selecionadas são escolhidas com base numa estratégia de seleção de ação.

    A aprendizagem tem um caráter temporal, ou seja, o resultado de uma ação só é
    conhecido após a sua execução. Por isso, a aprendizagem é feita em dois passos:
    1. Selecionar uma ação
    2. Atualizar a memória de aprendizagem com o resultado da ação
    """

    def __init__(self, acoes, memoria, sel_acao, metodo):
        self.__acoes = acoes
        self.__memoria = memoria
        self.__sel_acao = sel_acao
        self.__metodo = metodo

    def aprender(self, estado, acao, reforco, estado_seguinte, acao_seguinte=None):
        """
        Para o mecanismo de aprendizagem, "aprender" significa
        selecionar uma ação e do seu resultado atualizar a memória de aprendizagem.

        A aprendizagem pode ser on-policy ou off-policy:

        - Uma aprendizagem on-policy usa a mesma seleção de ação para aprender e
        para executar. A vantagem desta abordagem é que a aprendizagem converge
        para a política ótima. A desvantagem é que a aprendizagem é lenta.

        - Por outro lado, uma aprendizagem off-policy maximiza a aprendizagem em relação
        à política ótima, mas não converge para a mesma. A vantagem é que a aprendizagem
        é mais rápida. A desvantagem é que a aprendizagem pode divergir.
        Off-policy não utiliza a ação seguinte, porque só procura maximizar o valor e
        não aprender com o verdadeiro resultado da ação.
        """
        # Delegação ao método de aprendizagem
        self.__metodo.aprender(estado, acao, reforco, estado_seguinte, acao_seguinte)

    def selecionar_acao(self, estado):
        """
        Seleciona uma ação com base no estado atual.
        """
        # Delegação à estratégia de seleção de ação
        return self.__sel_acao.selecionar_acao(estado)

    def gerar_politica(self):
        """
        Gera uma política a partir da memória de aprendizagem.
        """
        return {
            estado: self.__sel_acao.aproveitar(estado)
            for estado in self.__memoria.estados
        }
