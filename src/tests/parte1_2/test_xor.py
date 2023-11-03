import numpy as np
from lib.rna.ativacao import Degrau
from lib.rna.camada import CamadaDensa
from lib.rna.rede_neuronal import RedeNeuronal

print("-- PROBLEMA XOR -- (cod. binária)")

# Definição da arquitetura da rede neuronal
rede = RedeNeuronal()

ativacao_degrau = Degrau(limiar=0)
camada_entrada = CamadaDensa(dim_entrada=0, dim_saida=2)
camada_escondida = CamadaDensa(
    dim_entrada=2, dim_saida=2, funcao_ativacao=ativacao_degrau
)
camada_saida = CamadaDensa(dim_entrada=2, dim_saida=1, funcao_ativacao=ativacao_degrau)

# Definição dos pesos e pendores das camadas
pesos_escondida = np.array([[1, -1], [-1, 1]])
pendores_escondida = np.array([-0.5, -0.5])
# camada_escondida.atualizar_pesos(pesos_escondida)
# camada_escondida.atualizar_pendores(pendores_escondida)

pesos_saida = np.array([[1], [1]])
pendor_saida = np.array([-0.5])
# camada_saida.atualizar_pesos(pesos_saida)
# camada_saida.atualizar_pendores(pendor_saida)

# Construção da rede neuronal
rede.juntar(camada_entrada)
rede.juntar(camada_escondida)
rede.juntar(camada_saida)

rede.atualizar_parametros(
    [(pesos_escondida, pendores_escondida), (pesos_saida, pendor_saida)]
)

# Aplicação da rede neuronal ao problema XOR
dados_entrada = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

rede.imprimir_previsao(dados_entrada)
# Resultado:
# [0 0] => [0.]
# [0 1] => [1.]
# [1 0] => [1.]
# [1 1] => [0.]
