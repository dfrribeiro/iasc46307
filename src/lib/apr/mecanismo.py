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
    Implementa os algoritmos de aprendizagem. Exemplos incluem SARSA e Q-Learning.

    A aprendizagem tem um caráter temporal, ou seja, o resultado de uma ação só é
    conhecido após a sua execução. Por isso, a aprendizagem é feita em dois passos:
    1) selecionar uma ação
    2) atualizar a memória de aprendizagem com o resultado da ação
    """

    def __init__(self, acoes):
        self.__acoes = acoes
        self.__memoria = None
        self.__metodo = None
        self.__sel_acao = None
        raise NotImplementedError

    def aprender(self, estado, acao, reforco, estado_seguinte, acao_seguinte=None):
        """
        Para o mecanismo de aprendizagem, "aprender" significa
        selecionar uma ação e do seu resultado atualizar a memória de aprendizagem.

        A aprendizagem pode ser on-policy ou off-policy:

        Uma aprendizagem on-policy usa a mesma seleção de ação para aprender e
        para executar. A vantagem desta abordagem é que a aprendizagem converge
        para a política ótima. A desvantagem é que a aprendizagem é lenta.

        Por outro lado, uma aprendizagem off-policy maximiza a aprendizagem em relação
        à política ótima, mas não converge para a mesma. A vantagem é que a aprendizagem
        é mais rápida. A desvantagem é que a aprendizagem pode divergir.
        Off-policy não utiliza a ação seguinte, porque só procura maximizar o valor e
        não aprender com o verdadeiro resultado da ação.
        """
        raise NotImplementedError

    def selecionar_acao(self, estado):
        raise NotImplementedError


# NOTAS

# politica:
# dicionario chave estado valor acao
# pi: determinista ou não determinista?
# politica otima: pi* = argmax Q*sa

# complexidade dos espaços de estados, tempo de convergência
# memória de experiencia, utilização de modelos do mundo, generalização/abstração
# (redes neuronais, modelos hierarquicos), arquiteturas híbridas

# memória de experiencia (n-passos. rasto de eligibilidade: acumular ou substituir)
# lambda - decaimento
# gamma - desconto temporal
