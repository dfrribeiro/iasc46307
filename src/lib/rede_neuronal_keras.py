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
        A função de perda utilizada é o erro quadrático médio.
        Para problemas de aprendizagem supervisionada, como é o caso, o erro quadrático
        médio é utilizado para medir a diferença entre as saídas desejadas e as saídas
        da rede neuronal.

        O otimizador utilizado é a descida de gradiente estocástico (SGD).
        A descida de gradiente estocástico é uma estratégia de otimização iterativa
        à procura de um mínimo local da função de perda. A cada iteração, o algoritmo
        atualiza os parâmetros da rede neuronal de acordo com a direção do gradiente.
        A taxa de aprendizagem controla o tamanho do passo dado em cada iteração.
        O momento controla a influência de iterações anteriores na direção do gradiente.
        A variação estocástica do gradiente é utilizada para evitar que o algoritmo
        fique preso em mínimos locais, inicializando cada iteração a partir de um ponto
        diferente.

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
