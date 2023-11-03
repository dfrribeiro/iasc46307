from .defamb import DEF_AMB
from .accao import Accao
from .elemento import Elemento

class Ambiente:
    """
    Simulador de ambiente 2D
    """

    def __init__(self, num_ambiente):
        """
        Construtor
        :param num_ambiente: número do ambiente
        """
        self._ambiente = DEF_AMB[num_ambiente]
        self._posicao_agente = self._obter_posicao_inicial()
        self.mostrar()

    @property
    def x_max(self):
        """
        Valor máximo de x
        """
        return len(self._ambiente[0])

    @property
    def y_max(self):
        """
        Valor máximo de y
        """
        return len(self._ambiente)
    
    def _obter_posicao_inicial(self):
        """
        Obter posição inicial do agente no ambiente
        :return: posição do inicial do agente
        """
        for y in range(self.y_max):
            for x in range(self.x_max):
                if self._obter_elemento((x, y)) == Elemento.AGENTE:
                    return (x, y)

    def reiniciar(self):
        """
        Reinicia a posição do agente no ambiente
        :return: posição do agente
        """
        self._mover_agente(self._obter_posicao_inicial())
        return self._posicao_agente

    def actuar(self, accao):
        """
        Executa uma acção no ambiente
        :param accao: acção a ser executada
        """
        x, y = self._posicao_agente
        if accao == Accao.NORTE:
            y -= 1
        elif accao == Accao.SUL:
            y += 1
        elif accao == Accao.ESTE:
            x += 1
        elif accao == Accao.OESTE:
            x -= 1
        # Mover agente se não houver colisão
        elem = self._obter_elemento((x, y))
        if elem != Elemento.OBSTACULO:
            self._mover_agente((x, y))

    def observar(self):
        """
        Observar o ambiente
        :return: posição do agente, elemento nessa posição
        """
        return self._posicao_agente, self._obter_elemento(self._posicao_agente)
    
    def _obter_elemento(self, posicao):
        """
        Obter elemento de uma posição do ambiente
        :param posicao: posição do elemento no ambiente
        :return: elemento na posição
        """
        x, y = posicao
        return Elemento(self._ambiente[y][x])
    
    def _mover_agente(self, posicao):
        """
        Alterar posição do agente
        :param posicao: nova posição do agente
        """
        self._posicao_agente = posicao

    def mostrar(self):
        """
        Mostrar ambiente na consola
        """
        for linha in self._ambiente:
            print(''.join(linha))
        print()

    def mostrar_politica(self, politica):
        """
        Mostrar política de acção no ambiente
        :param politica: política a mostrar
        """
        for y in range(self.y_max):
            for x in range(self.x_max):
                accao = politica.get((x, y))
                if accao is not None:
                    print(accao.value, end='')
                else:
                    print('.', end='')
            print()
        print()
