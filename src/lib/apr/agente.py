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

    def __init__(self, ambiente, mecanismo):
        self._mecanismo = mecanismo
        self._estado_terminal = False
        self._estado_atual = None
        self._acao_executada = None

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

    def _gerar_reforco(self, percepcao, VALOR_MAX=1, CUSTO_MOV=0.1):
        """
        Define o resultado no domínio do problema.
        """

        if percepcao.elemento == Elemento.ALVO:
            return VALOR_MAX
        elif percepcao.colisao:
            return -VALOR_MAX
        return CUSTO_MOV

    def _atuar(self, acao):
        """
        Descodifica a ação e atua no ambiente com a ação selecionada.
        """

        # TODO: descodificar acao
        acao_descodificada = Accao.NORTE
        self._ambiente.actuar(acao_descodificada)

        self._acao_executada = acao

    def executar(self):
        """
        Um passo do ciclo de aprendizagem do agente.
        """

        percepcao = self._percepcionar()
        acao = self._processar(percepcao)
        self._atuar(acao)
        self._ambiente.mostrar()

    def iniciar_episodio(self):
        """
        Um episodio de raciocinio no ambiente é um ciclo de aprendizagem
        que termina quando o agente atinge o estado objetivo.
        """
        try:
            while True:  # cancelado por CTRL+C/SIGINT
                estado = self._ambiente.reiniciar()

                # TODO: Métricas
                recompensa_total = 0
                step = 0

                while not self.estado_terminal:
                    step += 1
                    self.executar()
        except KeyboardInterrupt:
            # TODO save/log
            print("Episódio terminado por interrupção do utilizador.")


# Diagrama de sequência:
# acao = mecanismo.selecionar_acao(s)
# ambiente.atuar(acao)
# posicao, elemento = ambiente.observar()
# r = self.gerar_reforco(elemento)
# mecanismo.aprender(s, acao, r, sn)
