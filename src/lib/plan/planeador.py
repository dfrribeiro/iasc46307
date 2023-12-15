from lib.plan.frente_onda import FrenteOnda


class PlaneadorFrenteOnda:
    """
    Classe que implementa o planeador Frente de Onda.
    """

    def __init__(self, modelo, gamma=0.98, valor_max=1):
        self.__modelo = modelo
        self.__frente_onda = FrenteOnda(gamma, valor_max)
        self.__V = None  # Estado, valor

    @property
    def V(self):
        return self.__V

    def planear(self, objetivos):
        """
        Cria o plano (define uma política estado-ação) para atingir os objetivos.
        Utiliza a dependência para gerar o valor de cada estado.

        Retorna um dicionário que associa cada estado (não objetivo)
        à ação com maior valor.
        """
        self.__V = self.__frente_onda.propagar_valor(self.__modelo, objetivos)
        politica = {
            estado: max(
                self.__modelo.A, key=lambda acao: self.__valor_acao(estado, acao)
            )
            for estado in self.__modelo.S
            if estado not in objetivos  # Não interessa associar ações aos objetivos
            # Não se aplica a apanhar vários objetivos, apenas um
        }
        return politica

    def __valor_acao(self, estado, acao):
        """
        Dado um estado e uma ação,
        descobre o estado seguinte,
        e retorna o valor imediato de realizar a ação nesse estado.
        Caso não exista na função de valor, retorna o menor valor possível.
        """
        estado_seguinte = self.__modelo.T(estado, acao)
        return self.__V.get(estado_seguinte, float("-inf"))
