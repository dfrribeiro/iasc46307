class FrenteOnda:
    """
    Classe que implementa o algoritmo de Frente de Onda.

    O algoritmo de Frente de Onda é um algoritmo de procura em espaço
    de estados em largura que determina a distância de cada estado para
    um conjunto de estados objetivo. O valor é propagado de forma iterativa
    para os estados adjacentes, resultando num gradiente de valores que representa
    a distância de cada estado para os estados objetivo.
    """

    def __init__(self, gamma, valor_max):
        self.__gamma = gamma
        self.__valor_max = valor_max

    def propagar_valor(self, modelo, objetivos):
        """
        Propaga o valor de um estado para os estados vizinhos.
        Retorna um dicionário V com os valores para cada estado.
        """
        V = {}
        frente = []

        # Inicializa o valor dos estados objetivos e os adiciona à frente de onda
        for s in objetivos:
            V[s] = self.__valor_max
            frente.append(s)

        while frente:
            # Remove o primeiro elemento da frente de onda
            s = frente.pop(0)

            # Para cada estado adjacente, calcula o valor
            for sn in self.__adjacentes(modelo, s):
                # Utiliza uma função de decaimento exponencial para propagar o valor
                v = V[s] * self.__gamma ** modelo.distancia(s, sn)

                # Se a onda tem maior intensidade, propaga o valor
                if v > V.get(sn, float("-inf")):
                    V[sn] = v
                    frente.append(sn)
        return V

    def __adjacentes(self, modelo, estado):
        """
        Retorna uma lista dos estados adjacentes ao estado dado.
        Para tal, é necessário iterar sobre todas as ações possíveis do modelo
        no estado dado e determinar a transição de estado para cada ação.
        A lista de estados adjacentes não pode incluir o próprio estado nem
        estados inválidos (colisões, que aparecem como None).
        A lista de estados adjacentes não deve conter estados repetidos.
        """
        adjacentes = []
        for acao in modelo.A:
            prox_estado = modelo.T(estado, acao)
            if (
                prox_estado is not None
                and prox_estado != estado
                and prox_estado not in adjacentes
            ):
                adjacentes.append(prox_estado)
        return adjacentes
