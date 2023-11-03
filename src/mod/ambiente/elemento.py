from enum import Enum

#__________________________________________________

class Elemento(Enum):
    """
    Elemento do ambiente
    """
    AGENTE = '@'
    ALVO = '+'
    OBSTACULO = '#'
    VAZIO = ' '
