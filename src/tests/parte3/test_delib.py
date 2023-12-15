import sys
from lib.plan.agente import AgenteFrenteOnda

AMBIENTE_POR_OMISSAO = 2

if __name__ == "__main__":
    num_amb = int(sys.argv[1]) or AMBIENTE_POR_OMISSAO
    # if num_amb not in DEF_AMB

    agente = AgenteFrenteOnda(num_amb)
    agente.executar()
