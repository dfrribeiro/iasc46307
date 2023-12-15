import time
from lib.plan.modelo2d import ModeloMundo2D
from lib.plan.planeador import PlaneadorFrenteOnda
from mod.ambiente.defamb import DEF_AMB
from lib.plan.visualizador import VisValorPol


class AgenteDelib:
    """
    Classe abstrata para todos os agentes deliberativos.
    Um agente deliberativo é um agente que toma decisões
    com base num modelo do mundo e num objetivo.

    Ao contrário de um agente reativo, que toma decisões
    com base apenas na perceção do mundo, um agente deliberativo
    tem um modelo do mundo que lhe permite tomar decisões
    com base em estados não diretamente observáveis.

    NÃO inclui estratégias de reconsideração, que podem alterar crenças não associadas
    ao objetivo atual.
    """

    def executar(self):
        """
        Implementa o processo de tomada de decisão do agente:

        1. Percepcionar o mundo
        2. Atualizar o modelo do mundo, crenças internas
        3. Deliberar o que fazer
            1. Gerar opções
            2. Selecionar opções
        4. Planear como fazer, gerando um plano de ação
        5. Executar plano de ação
        """

        percepcao = self._percepcionar()
        self._atualizar_crencas(percepcao)
        objetivos = self._deliberar()
        plano = self._planear(objetivos)
        self._executar_plano(plano)

    def _percepcionar(self):
        """
        Método abstrato que define o processo de perceção.
        Retorna uma percepção.
        """
        raise NotImplementedError

    def _atualizar_crencas(self, percepcao):
        """
        Método abstrato que define o processo de atualização das crenças internas.
        Afeta objetos em memória e não retorna nada.
        """
        raise NotImplementedError

    def _deliberar(self):
        """
        Método abstrato que define o processo de deliberação.
        Retorna lista de estados objetivo.
        """
        raise NotImplementedError

    def _planear(self, objetivos):
        """
        Método abstrato que define o processo de planeamento.
        Recebe lista de estados objetivo, retorna plano.
        """
        raise NotImplementedError

    def _executar_plano(self, plano):
        """
        Método abstrato que define o processo de execução.
        Recebe plano, não retorna nada.
        """
        raise NotImplementedError


class AgenteFrenteOnda(AgenteDelib):
    """
    Classe que implementa um agente deliberativo
    que utiliza o algoritmo de Frente de Onda.
    """

    def __init__(self, num_ambiente):
        self.__num_ambiente = num_ambiente
        self.__modelo = ModeloMundo2D()
        self.__planeador = PlaneadorFrenteOnda(self.__modelo)
        self.__visualizador = VisValorPol()

    def _percepcionar(self):
        """
        Obtém a definição do ambiente: dicionário de listas de elementos (símbolos)

        É como se a perceção retornasse uma imagem do mundo.
        """

        return DEF_AMB[self.__num_ambiente]

    def _atualizar_crencas(self, percepcao):
        """
        As "crenças" do agente estão representadas no modelo do mundo:
        são as posições dos obstáculos e dos alvos oriundas da perceção.
        """
        self.__modelo.atualizar(percepcao)

    def _deliberar(self):
        """
        Que estados o agente pretende atingir? (no modelo do mundo)
        """
        return self.__modelo.obter_posicoes_alvo()

    def _planear(self, objetivos):
        """
        Utiliza o planeador para gerar um plano de ação
        """
        return self.__planeador.planear(objetivos)

    def _executar_plano(self, plano):
        """
        Visualiza o plano (não mexe o agente)
        """
        now = time.strftime("%Y%m%d-%H%M%S")

        self.__visualizador.mostrar(
            self.__modelo.xmax,
            self.__modelo.ymax,
            self.__planeador.V,
            plano,
            nome_ficheiro=f"out/plan_{self.__num_ambiente}_{now}.png",
        )
