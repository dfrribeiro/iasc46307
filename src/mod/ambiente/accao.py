from enum import Enum

#__________________________________________________

class Accao(Enum):
    """
    Acção de movimentação no ambiente
    (representação com caracteres unicode)
    """
    NORTE = '\u2191'
    SUL = '\u2193'
    ESTE = '\u2192'
    OESTE = '\u2190'
