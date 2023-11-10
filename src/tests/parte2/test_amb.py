from mod.ambiente.ambiente import Ambiente
from mod.ambiente.accao import Accao
from lib.apr.agente import Agente
from lib.apr.mecanismo import MecanismoAprendizagem


class TesteAprendRef:
    """
    Classe que modulariza o teste de aprendizagem por reforço.
    """

    def testar(self, num_ambiente, num_episodios):
        """
        Retorna o número de passos por episódio.
        """

        # Definição do ambiente
        ambiente = Ambiente(num_ambiente)

        # Definição do mecanismo de aprendizagem
        acoes = list(Accao)
        mec_aprend = MecanismoAprendizagem(acoes)

        # Definição do agente
        agente = Agente(ambiente, mec_aprend)

        # Ciclo de aprendizagem do agente
        return agente.executar(num_episodios)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time

    NUM_AMBIENTE = 1
    MAX_EPISODIOS = 100

    teste = TesteAprendRef()
    passos = teste.testar(NUM_AMBIENTE, MAX_EPISODIOS)

    # Gráfico dos resultados
    plt.plot(passos, ".--", label="Q-Learning")
    plt.title(f"Ambiente #{NUM_AMBIENTE}: Número de passos por episódio")
    plt.xlabel("Episódio")
    plt.ylabel("Passos")
    plt.grid(True)
    plt.legend()
    now = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig(f"out/apr_{NUM_AMBIENTE}_{now}.png")
    plt.show()
