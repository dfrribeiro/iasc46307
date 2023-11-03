class MemoriaAprendizagem:
    """
    A memória de aprendizagem é um dicionário que guarda as associações entre
    (estado, ação) e valor.
    """

    def __init__(self):
        raise NotImplementedError

    def atualizar(self, estado, acao, valor):
        raise NotImplementedError

    def obter_valor(self, estado, acao):
        """
        Função de valor Q
        """
        raise NotImplementedError


class MemoriaEsparsa(MemoriaAprendizagem):
    """
    Uma memória de aprendizagem esparsa é uma memória que guarda apenas os valores
    não nulos, e devolve um valor por omissão para os restantes.
    """

    def __init__(self, valor_por_omissao=0.0):
        self.q = {}
        self.__valor_por_omissao = valor_por_omissao

    def atualizar(self, estado, acao, valor):
        self.q[estado, acao] = valor

    def obter_valor(self, estado, acao):
        return self.q.get((estado, acao), self.__valor_por_omissao)

    def __getitem__(self, index):
        return self.obter_valor(*index)

    def __setitem__(self, index, value):
        self.atualizar(*index, value)


"""
Acelerar o processo de aprendizagem:
Replay memory - memória de experiência
Tamanho n limitado pela capacidade computacional
Limitar memória - eliminar experiências antigas ou menos relevantes
Experiencia = (s, a, r_, s_)
Modelo do mundo: estrutura, dinâmica, valor - suporta planeamento, simula experiência
"""

if __name__ == "__main__":
    # Testes unitários
    memoria = MemoriaEsparsa()
    memoria.atualizar(1, "<", 1)
    memoria.atualizar(1, ">", 2)
    memoria.atualizar(2, "<", 4)
    memoria.atualizar(2, ">", 3)

    print(memoria[1, "<"])
    print(memoria[1, ">"])
