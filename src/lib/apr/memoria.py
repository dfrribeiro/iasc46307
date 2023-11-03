import numbers


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
        Função de valor
        """
        raise NotImplementedError


class MemoriaEsparsa(MemoriaAprendizagem):
    """
    Uma memória de aprendizagem esparsa é uma memória que guarda apenas os valores
    não nulos, e devolve um valor por omissão para os restantes.
    Utiliza-se uma memória esparsa quando o espaço de estados não está completamente
    definido, ou seja, quando não se sabe de antemão todos os estados possíveis.
    Também pode ser utilizada para poupar memória.
    """

    def __init__(self, valor_por_omissao=0.0):
        self.__entradas = MemoriaAssociativa()
        self.__valor_por_omissao = valor_por_omissao

    def atualizar(self, estado, acao, valor):
        self.__entradas[estado, acao] = valor

    def obter_valor(self, estado, acao):
        return self.__entradas.get((estado, acao), self.__valor_por_omissao)

    def __getitem__(self, index):
        return self.obter_valor(*index)

    def __setitem__(self, index, value):
        self.atualizar(*index, value)

    def __str__(self):
        linhas = []
        for (estado, acao), valor in self.__entradas.items():
            linhas.append(f"({estado}, {acao}): {valor}")
        linhas += [f"(?, ?): {self.__valor_por_omissao}"]

        return "\n".join(linhas)


class MemoriaAssociativa(dict):
    """
    Uma memória associativa é um mapa de (estado, ação) para valor,
    acessível pela chave.
    No caso Python é um simples dicionário mas pode ser implementado de forma
    diferente como por exemplo uma HashTable na linguagem C.
    A memória associativa garante que as suas chaves são sempre tuplos de dois
    elementos e que o seu valor é sempre um número.
    """

    def __init__(self):
        super().__init__()

    def __setitem__(self, chave, valor):
        if not self.__validar_chave(chave):
            raise ValueError("Chave deve ser um tuplo de dois elementos (s, a)")
        if not self.__validar_valor(valor):
            raise ValueError("Valor deve ser um número")
        super().__setitem__(chave, valor)

    def __getitem__(self, chave):
        if not self.__validar_chave(chave):
            raise KeyError("Chave deve ser um tuplo de dois elementos (s, a)")
        return super().__getitem__(chave)

    def get(self, chave, default=None):
        if not self.__validar_chave(chave):
            raise KeyError("Chave deve ser um tuplo de dois elementos (s, a)")
        return super().get(chave, default)

    @staticmethod
    def __validar_chave(chave):
        return isinstance(chave, tuple) and len(chave) == 2

    @staticmethod
    def __validar_valor(valor):
        return isinstance(valor, numbers.Number)


"""
Acelerar o processo de aprendizagem:
Replay memory - memória de experiência
Tamanho n limitado pela capacidade computacional
Limitar memória - eliminar experiências antigas ou menos relevantes
Experiencia = (s, a, r_, s_)
Modelo do mundo: estrutura, dinâmica, valor - suporta planeamento, simula experiência
"""

if __name__ == "__main__":
    # Testes
    memoria = MemoriaEsparsa()
    memoria.atualizar(1, "<", 1)
    memoria.atualizar(1, ">", 2)
    memoria.atualizar(2, "<", 4)
    memoria.atualizar(2, ">", 3)

    print(memoria)

    assert memoria[1, "<"] == 1
    assert memoria[1, ">"] == 2
