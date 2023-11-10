import random


class EstrategiaSelecaoAcao:
    """
    Uma estratégia de seleção de ação é um objeto que encapsula a lógica
    de seleção de ação.
    """

    def __init__(self, memoria):
        self._memoria = memoria

    def selecionar_acao(self, estado):
        raise NotImplementedError


class EpsilonGreedy(EstrategiaSelecaoAcao):
    """
    A estratégia de seleção de ação Epsilon-Greedy determina que o
    agente escolhe uma ação aleatória, ignorando o seu valor aprendido,
    numa percentagem epsilon das suas seleções.

    A exploração é importante para obter melhorias a uma solução inicial que
    o agente tenha obtido.

    Um valor de epsilon de 1 resulta num agente irracional,
    que escolhe as ações aleatoriamente.
    """

    def __init__(self, memoria, acoes, epsilon, seed=None):
        super().__init__(memoria)
        self.__acoes = acoes
        self.__epsilon = epsilon
        random.seed(seed)

    def selecionar_acao(self, estado):
        """
        O agente calcula um valor entre 0 e 1. Se este valor calhar no intervalo
        [0, epsilon[, o agente escolhe explorar em vez de aproveitar.
        """
        return (
            self.explorar()
            if random.random() < self.__epsilon
            else self.aproveitar(estado)
        )

    def aproveitar(self, estado):
        """
        Aproveitar significa maximizar o valor da ação para o estado dado.
        O quão efetivo esta estratégia é depende do conhecimento colecionado
        em memória. Uma memória com pouco conhecimento é mais provável de escolher
        uma ação que já tenha explorado ou ainda não tenha explorado, dependendo da
        relação entre reforços e valores iniciais de Q, do que a ação ótima para o
        problema dado.
        """
        return self.acao_sofrega(estado)

    def explorar(self):
        """
        O agente escolhe uma ação aleatória a partir de todas as ações possíveis.
        Note-se que mesmo explorando é possível que o agente escolha a ação
        com maior valor na sua memória, por mero acaso. Por esta razão, o valor
        de epsilon não representa necessariamente a frequência com que se "desvia
        do percurso".
        """
        return random.choice(self.__acoes)

    def acao_sofrega(self, estado):
        """
        Seleciona a ação com maior valor Q para o estado dado.
        Baralha as ações para evitar favoritismo.
        """
        random.shuffle(self.__acoes)
        return max(self.__acoes, key=lambda a: self._memoria[estado, a])


if __name__ == "__main__":
    from math import ceil

    # Testes unitários
    acoes = ["<", ">"]
    memoria = {(1, "<"): 1, (1, ">"): 2, (2, "<"): 4, (2, ">"): 3}

    epsilon = 0.25  # 1 em cada 4 ações são aleatórias
    esperadas_acoes_aleatorias_por_teste = 3
    sel_acao = EpsilonGreedy(memoria, acoes, epsilon)

    print("Estado 1:")
    print(
        *[
            sel_acao.selecionar_acao(1)
            for _ in range(ceil(esperadas_acoes_aleatorias_por_teste / epsilon))
        ],
    )

    print("Estado 2:")
    print(
        *[
            sel_acao.selecionar_acao(2)
            for _ in range(ceil(esperadas_acoes_aleatorias_por_teste / epsilon))
        ],
    )
