class RedeNeuronal:
    """
    Classe para representar uma rede neuronal.

    Parâmetros:
        camadas: Lista de camadas da rede neuronal.

    """

    def __init__(self):
        self.camadas = []

    def juntar(self, camada):
        """
        Junta uma camada à rede neural.

        Parâmetros:
            camada: Camada a ser adicionada à rede neural.

        Exceções:
            AssertionError: Se a dimensão de entrada da nova camada não for igual à
            dimensão de saída da última camada, ou se a dimensão de entrada da primeira
            camada não for 0.
        """

        if len(self.camadas) == 0:
            assert camada.dim_entrada == 0
        else:
            assert camada.dim_entrada == self.camadas[-1].dim_saida

        self.camadas.append(camada)

    def prever(self, entradas):
        """
        Realiza a previsão da rede neuronal para as entradas fornecidas.

        Parâmetros:
            entradas: Entradas da rede neuronal.

        Retorna:
            Saídas da rede neuronal depois de ativadas todas as camadas (feedforward).

        """

        for camada in self.camadas:
            entradas = camada.ativar(entradas)

        return entradas

    def atualizar_parametros(self, parametros):
        """
        Atualiza os pesos e pendores de todas as camadas da rede neuronal exceto a
        camada de entrada.

        Parâmetros:
            parametros: Lista de tuplos com os pesos e pendores de cada camada.
        """

        assert len(parametros) == len(self.camadas) - 1
        for i, (pesos, pendores) in enumerate(parametros):
            self.camadas[i + 1].atualizar_pesos(pesos)
            self.camadas[i + 1].atualizar_pendores(pendores)

    def imprimir_previsao(self, entradas):
        """
        Realiza a previsão para as entradas fornecidas e imprime os resultados
        no formato 'entrada => saída'.

        Args:
            entradas: Os dados de entrada da rede neuronal.

        Retorna:
            Saídas da rede neuronal, caso queiram ser utilizadas para outros fins.

        Exemplo:
        >>> rede_neuronal = RedeNeuronal()
        >>> entradas = [[0.1,], [0.2,], [0.3,]]
        >>> rede_neuronal.imprimir_previsao(entradas)
        Previsão:
        0.1 => 0.5
        0.2 => 0.6
        0.3 => 0.7
        """
        saidas = self.prever(entradas)
        print("Previsão:")
        [print(f"{elemento[0]} => {elemento[1]}") for elemento in zip(entradas, saidas)]
        return saidas
