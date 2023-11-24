from lib.apr.estrategia import EpsilonGreedy
from lib.apr.memoria import MemoriaEsparsa
from lib.apr.metodo import QLearning
from lib.apr.qme import QME
from mod.ambiente.ambiente import Ambiente
from mod.ambiente.accao import Accao
from lib.apr.agente import Agente
from lib.apr.mecanismo import MecanismoAprendizagem

DEFAULT_CONFIG = {
    "memoria": {"class": MemoriaEsparsa, "args": {}},
    "sel_acao": {"class": EpsilonGreedy, "args": {"epsilon": 0.1}},
    "metodo": {"class": QLearning, "args": {"alpha": 0.5, "gamma": 0.9}},
    "mec_aprend": {"class": MecanismoAprendizagem, "args": {}},
}


def definir_agente(config):
    base = DEFAULT_CONFIG
    base.update(config)
    return base


def construir_agente(ambiente, acoes, config):
    """
    Constrói um agente com base na configuração dada.
    Assume que a configuração provém de `definir_agente`.
    """
    m, s, t, c = (
        config["memoria"],
        config["sel_acao"],
        config["metodo"],
        config["mec_aprend"],
    )

    # Definição do agente (inicialização de dependências)
    # + com configuração por omissão
    memoria = m["class"](**m["args"])
    sel_acao = s["class"](memoria, acoes, **s["args"])
    metodo = t["class"](memoria, sel_acao, **t["args"])
    mec_aprend = c["class"](acoes, memoria, sel_acao, metodo, **c["args"])

    return Agente(ambiente, mec_aprend)  # reforço max = 100


def mostrar_agente(agente):
    def pretty_args(d):
        return ", ".join(f"{k}={v}" for k, v in d.items()) if d.items() else ""

    print(f"Agente: {agente['label']}")
    print(
        f"| Memória: {agente['memoria']['class'].__name__}\
({pretty_args(agente['memoria']['args'])})"
    )
    print(
        f"| Seleção de ação: {agente['sel_acao']['class'].__name__}\
({pretty_args(agente['sel_acao']['args'])})"
    )
    print(
        f"| Método de aprendizagem: {agente['metodo']['class'].__name__}\
({pretty_args(agente['metodo']['args'])})"
    )
    print(
        f"| Mecanismo de aprendizagem: {agente['mec_aprend']['class'].__name__}\
({pretty_args(agente['mec_aprend']['args'])})\n"
    )


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time

    NUM_AMBIENTE = 2
    MAX_EPISODIOS = 50

    # Definição declarativa dos agentes
    config_agentes = [
        {
            "label": "Q-Learning",
        },
        {
            "label": "Q-Learning com Memória de Experiência",
            "metodo": {
                "class": QME,
                "args": {
                    "alpha": 0.5,
                    "gamma": 0.9,  # TODO manter por omissão
                    "num_sim": 5,
                    "dim_max": 50,
                },
            },
        },
        {
            "label": "Q-Learning otimista",
            "memoria": {
                "class": MemoriaEsparsa,
                "args": {"valor_por_omissao": 10},
                # Qualquer valor acima de zero é otimista, visto que o agente não
                # perde nada por explorar. O valor 10 é arbitrário.
            },
        },
    ]

    acoes = list(Accao)

    for num_ambiente in range(1, 4):
        ambiente = Ambiente(num_ambiente)

        for i, config in enumerate(config_agentes):
            # Definição do agente
            config["label"] = config.get("label") or f"Agente #{i}"
            def_agente = definir_agente(config)
            mostrar_agente(def_agente)

            agente = construir_agente(ambiente, acoes, def_agente)

            # Gráfico dos resultados
            plt.plot(agente.executar(MAX_EPISODIOS), ".--", label=config["label"])

            # Mostrar política (Q) aprendida
            # s: argmax_a Q(s, a) for s in estados
            politica = agente._mecanismo.gerar_politica()
            ambiente.mostrar_politica(politica)

        plt.title(f"Ambiente #{num_ambiente}: Número de passos por episódio")
        plt.xlabel("Episódio")
        plt.ylabel("Passos")
        plt.grid(True)
        plt.legend()
        now = time.strftime("%Y%m%d-%H%M%S")
        plt.savefig(f"out/apr_{num_ambiente}_{now}.png")
        plt.show()
