from mod.ambiente.ambiente import Ambiente
from lib.apr.agente import Agente
from lib.apr.mecanismo import MecanismoAprendizagem

if __name__ == "__main__":
    # Definição do ambiente
    ambiente = Ambiente(1)
    ambiente.mostrar()

    # Definição do mecanismo de aprendizagem
    mec_aprend = MecanismoAprendizagem()

    # Definição do agente
    agente = Agente(ambiente, mec_aprend)

    # Ciclo de aprendizagem do agente
    agente.executar()
