from keras.models import Sequential
from keras.optimizers import SGD


class RedeNeuronal:
    """
    Encapsulamento da interface Keras para uma rede neuronal.
    """

    def __init__(self):
        self.__modelo = Sequential()

    def juntar(self, camada):
        """
        Adiciona uma camada à rede neuronal.

        Parâmetros:
            camada: Camada a adicionar, compatível com modelos sequenciais Keras.

        """

        self.__modelo.add(camada)

    def prever(self, entradas):
        """
        Executa a rede neuronal para uma dada entrada.

        Parâmetros:
            entradas: Entradas da rede neuronal.

        Retorna:
            Saídas da rede neuronal.

        """

        return self.__modelo.predict(entradas)

    def treinar(
        self,
        entradas,
        saidas,
        epocas,
        taxa_aprendizagem,
        momento=0.0,
        ordem_aleatoria=False,
    ):
        """
        Treina a rede neuronal utilizando o algoritmo de retropropagação.

        Parâmetros:
            entradas: Entradas da rede neuronal.
            saidas: Saídas desejadas para as entradas fornecidas.
            epocas: Número de épocas de treino.
            taxa_aprendizagem: Taxa de aprendizagem.
            momento: Momento.
            ordem_aleatoria: Se verdadeiro, as entradas são apresentadas à rede neuronal
            por ordem aleatória.

        Retorna:
            Lista com os erros de cada época.

        """

        self.__modelo.compile(
            loss="mean_squared_error",
            optimizer=SGD(learning_rate=taxa_aprendizagem, momentum=momento),
        )

        return self.__modelo.fit(
            entradas, saidas, epochs=epocas, verbose=0, shuffle=ordem_aleatoria
        ).history["loss"]

    def mostrar(self):
        """
        Mostra a estrutura da rede neuronal,
        no formato de tabela (camada, dimensão, parâmetros).

        """

        self.__modelo.summary()
