# from mod.ambiente.accao import Accao
from mod.ambiente.accao import Accao
from mod.ambiente.elemento import Elemento
from collections import namedtuple

# Representações internas
Percepcao = namedtuple("Percepcao", ["posicao", "alvo", "colisao"])
Posicao = namedtuple("Posicao", ["x", "y"])
# Espaço de ação - enumeração (valores discretos)


class Agente:
    """
    Classe base para todos os agentes.
    Um agente é um objeto que interage com o ambiente.
    O agente age no ambiente através de ações.
    As ações são selecionadas com base na percepção do ambiente.
    Para aprender, o agente tem um mecanismo de aprendizagem.
    """

    def __init__(self, ambiente, mecanismo, recompensa_max=100.0):
        self._mecanismo = mecanismo
        self._estado_atual = None
        self._acao_executada = None
        self._recompensa_max = recompensa_max

        self._associar_ambiente(ambiente)

    def _associar_ambiente(self, ambiente):
        self._ambiente = ambiente
        self._estado_atual = self._ambiente.reiniciar()

    def _percepcionar(self):
        """
        Codifica o estado do ambiente
        """
        posicao, elemento = self._ambiente.observar()
        alvo = elemento == Elemento.ALVO
        colisao = self._estado_atual == posicao or elemento == Elemento.OBSTACULO
        return Percepcao(posicao, alvo, colisao)

    def _processar(self, percepcao):
        """
        Seleciona a ação e atualiza o mecanismo de aprendizagem
        """
        acao_selecionada = self._mecanismo.selecionar_acao(percepcao.posicao)
        reforco = self._gerar_reforco(percepcao)

        # TODO: estado atual ou percepcao.posicao? colisao?
        estado_seguinte = self._estado_atual.aplicar(acao_selecionada)
        self._mecanismo.aprender(
            self._estado_atual, acao_selecionada, reforco, estado_seguinte
        )

        return acao_selecionada

    @property
    def estado_terminal(self):
        return self._ambiente._obter_elemento(self._estado_atual) == Elemento.ALVO

    def _gerar_reforco(self, percepcao, CUSTO_MOV=0.1):
        """
        Define o resultado no domínio do problema.
        """

        if percepcao.elemento == Elemento.ALVO:
            return self._recompensa_max
        elif percepcao.colisao:
            return -self._recompensa_max
        return -CUSTO_MOV

    def _atuar(self, acao):
        """
        Descodifica a ação e atua no ambiente com a ação selecionada.
        """

        # TODO: descodificar acao
        acao_descodificada = Accao.NORTE
        self._ambiente.actuar(acao_descodificada)

        self._acao_executada = acao

    def _passo_episodio(self):
        """
        Um passo do ciclo de aprendizagem do agente.
        """

        percepcao = self._percepcionar()
        acao = self._processar(percepcao)
        self._atuar(acao)
        self._ambiente.mostrar()

    def _fim_episodio(self):
        return self._ambiente._obter_elemento(self._estado_atual) == Elemento.ALVO

    def executar(self, num_episodios=None):
        """
        Um episodio de raciocinio no ambiente é um ciclo de aprendizagem
        que termina quando o agente atinge o estado objetivo.
        Retorna uma lista com o número de passos que o agente
        demorou a atingir o objetivo em cada episódio.
        Se num_episodios for None, executa até o utilizador interromper.
        """
        num_passos_episodio = []
        episodio = 0
        try:
            while num_episodios is None or episodio < num_episodios:
                episodio += 1
                self._estado_atual = self._ambiente.reiniciar()

                num_passos = 0
                while not self._fim_episodio():
                    num_passos += 1
                    self._passo_episodio()

                # Registar o número de passos que o agente demorou a atingir o objetivo
                num_passos_episodio.append(num_passos)
        except KeyboardInterrupt:
            print(
                f"Execução interrompida pelo utilizador.\
                \nNº de episódios executados: {episodio}"
            )
        return num_passos_episodio


# Diagrama de sequência:
# acao = mecanismo.selecionar_acao(s)
# ambiente.atuar(acao)
# posicao, elemento = ambiente.observar()
# r = self.gerar_reforco(elemento)
# mecanismo.aprender(s, acao, r, sn)
