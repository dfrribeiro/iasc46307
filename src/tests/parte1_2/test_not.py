import numpy as np
from lib.ativacao import Degrau
from lib.camada import CamadaDensa
from lib.rede_neuronal import RedeNeuronal

print("-- PROBLEMA NOT -- (cod. binária)")

# Definição da arquitetura da rede neuronal
rede = RedeNeuronal()

ativacao_degrau = Degrau(limiar=0)
camada_entrada = CamadaDensa(dim_entrada=0, dim_saida=1)
camada_saida = CamadaDensa(dim_entrada=1, dim_saida=1, funcao_ativacao=ativacao_degrau)

# Definição dos pesos e pendores da camada
pesos = np.array(
    [
        [
            -1,
        ]
    ]
)
pendor = np.array(
    [
        0.5,
    ]
)

# Construção da rede neuronal
rede.juntar(camada_entrada)
rede.juntar(camada_saida)

rede.atualizar_parametros(
    [
        (pesos, pendor),
    ]
)

# Aplicação da rede neuronal ao problema NOT
dados_entrada = np.array(
    [
        [
            0,
        ],
        [
            1,
        ],
    ]
)

rede.imprimir_previsao(dados_entrada)
