from mod.ambiente.elemento import Elemento


class Agente:
    """
    Classe base para todos os agentes.

    Um agente é um objeto que interage com o ambiente.
    O agente age no ambiente através de ações.
    As ações são selecionadas com base na percepção do ambiente.
    Para se adaptar ao ambiente, o agente tem um mecanismo de aprendizagem.
    """

    def __init__(self, ambiente, mecanismo, reforco_max=100.0):
        self._mecanismo = mecanismo
        self._acao_selecionada = None
        self._reforco_max = reforco_max

        # O agente tem de conhecer o ambiente para poder interagir com ele
        self._ambiente = ambiente
        self._estado_atual = self._ambiente.reiniciar()

    def _passo_episodio(self):
        """
        Em cada passo, o agente:
        1. Seleciona a ação baseado no estado atual
        2. Atua no ambiente
        3. Observa o estado seguinte
        4. Calcula o reforço
        5. Aprende com o reforço
        6. Atualiza o estado atual
        """
        self._acao_selecionada = self._mecanismo.selecionar_acao(self._estado_atual)
        self._ambiente.actuar(self._acao_selecionada)
        estado_seguinte, elemento_seguinte = self._ambiente.observar()
        reforco = self._gerar_reforco(
            elemento_seguinte, self._estado_atual, estado_seguinte
        )

        self._mecanismo.aprender(
            self._estado_atual, self._acao_selecionada, reforco, estado_seguinte
        )
        self._estado_atual = estado_seguinte

    def _gerar_reforco(self, elemento_seguinte, estado_atual, estado_seguinte):
        """
        Define o resultado no domínio do problema.

        É habitual que o reforço por omissão tenha um valor negativo correspondente ao
        custo do movimento, para incentivar o agente a terminar o episódio o mais
        rapidamente possível.
        """
        if elemento_seguinte == Elemento.ALVO:
            return self._reforco_max
        elif elemento_seguinte == Elemento.OBSTACULO:
            return -self._reforco_max
        return 0

    def _fim_episodio(self):
        _, elemento = self._ambiente.observar()
        return elemento == Elemento.ALVO

    def executar(self, num_episodios=None):
        """
        Um episodio de raciocínio no ambiente é um ciclo de aprendizagem
        que termina quando o agente atinge o estado objetivo.

        Se num_episodios for None, executa até o utilizador interromper.

        Retorna uma lista com o número de passos que o agente
        demorou a atingir o objetivo em cada episódio.
        """
        num_passos_episodio = []
        episodio = 0
        try:
            while num_episodios is None or episodio < num_episodios:
                episodio += 1
                self._estado_atual = self._ambiente.reiniciar()

                # Avançar até o agente atingir o estado objetivo
                num_passos = 0
                while not self._fim_episodio():
                    self._passo_episodio()
                    num_passos += 1

                # Registar o número de passos que o agente demorou a atingir o objetivo
                num_passos_episodio.append(num_passos)
        except KeyboardInterrupt:
            print(
                f"Execução interrompida pelo utilizador.\
                \nNº de episódios executados: {episodio}"
            )
        return num_passos_episodio
