from lib.apr.estrategia import EpsilonGreedy
from lib.apr.memoria import MemoriaEsparsa
from lib.apr.metodo import QLearning
from lib.apr.qme import QME
from mod.ambiente.ambiente import Ambiente
from mod.ambiente.accao import Accao
from lib.apr.agente import Agente
from lib.apr.mecanismo import MecanismoAprendizagem


class TesteAprendRef:
    """
    Classe que modulariza o teste de aprendizagem por reforço.
    """

    # TODO: extrair agente para permitir comparar qualquer agente (separar escopos)
    def testar(self, num_ambiente, num_episodios):
        """
        Retorna o número de passos por episódio.
        """

        # Definição do ambiente
        ambiente_q = Ambiente(num_ambiente)
        ambiente_qme = Ambiente(num_ambiente)

        # Definição do agente para os dois métodos de aprendizagem
        acoes_q = list(Accao)
        memoria_q = MemoriaEsparsa()
        sel_acao_q = EpsilonGreedy(memoria_q, acoes_q, epsilon=0.1)
        metodo_q = QLearning(memoria_q, sel_acao_q, alpha=0.5, gamma=0.9)
        mec_aprend_q = MecanismoAprendizagem(acoes_q, memoria_q, sel_acao_q, metodo_q)
        agente_q = Agente(ambiente_q, mec_aprend_q)

        #
        acoes_qme = list(Accao)
        memoria_qme = MemoriaEsparsa()
        sel_acao_qme = EpsilonGreedy(memoria_qme, acoes_qme, epsilon=0.1)
        metodo_qme = QME(
            memoria_qme,
            sel_acao_qme,
            alpha=0.5,
            gamma=0.9,
            num_sim=5,
            dim_max=50,
        )
        mec_aprend_qme = MecanismoAprendizagem(
            acoes_qme, memoria_qme, sel_acao_qme, metodo_qme
        )
        agente_qme = Agente(ambiente_qme, mec_aprend_qme)

        # Ciclo de aprendizagem do agente
        return [agente_q.executar(num_episodios), agente_qme.executar(num_episodios)]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time

    NUM_AMBIENTE = 1
    MAX_EPISODIOS = 100

    teste = TesteAprendRef()
    passos = teste.testar(NUM_AMBIENTE, MAX_EPISODIOS)

    # Gráfico dos resultados
    plt.plot(passos[0], ".--", label="Q-Learning")
    plt.plot(passos[1], ".--", label="Q-Learning com Memória de Experiência")
    plt.title(f"Ambiente #{NUM_AMBIENTE}: Número de passos por episódio")
    plt.xlabel("Episódio")
    plt.ylabel("Passos")
    plt.grid(True)
    plt.legend()
    now = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig(f"out/apr_{NUM_AMBIENTE}_{now}.png")
    plt.show()
