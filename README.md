# Projeto Incremental

|  |  |
|------------------------|-----------------|
| **Unidade Curricular** | Inteligência Artificial e Sistemas Cognitivos |
| **Docente**            | Luís Morgado    |
| **Semestre**           | Inverno 2023/24 |
| **Filiação**           | ISEL-DEETC      |

## Temas Principais

- Introdução à inteligência artificial
- Sistemas cognitivos, cognição e racionalidade, racionalidade limitada, meta-cognição
- Memória, adaptação e aprendizagem, redes neuronais artificiais, aprendizagem por reforço, algoritmos genéticos
- Raciocínio automático e tomada de decisão, raciocínio para planeamento, otimização e decisão sequencial, raciocínio prático, raciocínio com recursos limitados
- Representação de conhecimento, espaços conceptuais, formação de conceitos, representações simbólicas e sub-simbólicas, modelos cognitivos, significado e inferência
- Arquiteturas cognitivas, arquiteturas reativas, deliberativas e híbridas, integração de níveis cognitivos
- Inteligência artificial distribuída, sistemas multi-agente, comunicação e coordenação, interação e raciocínio social

## Instalação e Execução

### Pré-requisitos
- Python >=3.6 instalado [(instruções)](https://www.python.org/downloads/).

### Instalação
```shell
# Copiar o repositório para local
git clone https://github.com/dfrribeiro/iasc46307.git

# Entrar na pasta do repositório
cd iasc46307

# Criar um ambiente virtual
python -m venv venv

# Ativar o ambiente virtual (Windows)
& venv\Scripts\activate

# Ativar o ambiente virtual (UNIX)
source venv/bin/activate

# Instalar as dependências
pip install -r requirements.txt

# Adicionar o diretório do projeto ao PYTHONPATH (Windows Command Prompt)
set PYTHONPATH=%PYTHONPATH%;%cd%\src

# Adicionar o diretório do projeto ao PYTHONPATH (Windows PowerShell)
$env:PYTHONPATH += ";$PWD/src"

# Adicionar o diretório do projeto ao PYTHONPATH (UNIX)
export PYTHONPATH=$PYTHONPATH:$PWD/src
```

### Execução

Por exemplo, para executar o programa da parte 1.1:

```shell
python ./src/tests/parte1_1/xor.py
```

## Histórico de Versões

### [v1.1] - 22 de setembro de 2023

**Contexto**: Estudo de redes neuronais artificiais

**Objetivo**: Implementação da função XOR com uma rede neuronal do tipo perceptrão

**Requisitos**:
- A rede deve ser representada com base em matrizes
- A implementação deve ser realizada na linguagem Python sem utilização de classes
- A implementação deve ser realizada de acordo com o estudado nas aulas

### [v1.2] - 29 de setembro de 2023

**Contexto**: Estudo de redes neuronais artificiais

**Objetivo**: Implementação de uma rede neuronal multicamada e testes dos operadores booleanos OR, AND e NOT, bem como do operador XOR

**Requisitos**:
- A rede deve ser representada com base numa classe designada RedeNeuronal
- A implementação deve ser realizada na linguagem Python
- A implementação deve ser realizada de acordo com o estudado nas aulas

### [v1.3] - 6 de outubro de 2023

**Contexto**: Estudo de redes neuronais artificiais

**Objetivo**: Implementação de uma rede neuronal multicamada com capacidade de aprendizagem usando a plataforma Keras, com estudo do operador XOR considerando diversos aspetos.

**Requisitos**:
- A rede deve ser representada com base numa classe designada RedeNeuronal
- A implementação deve ser realizada na linguagem Python
- A implementação deve ser realizada de acordo com o estudado nas aulas

***Alterações***:
- Melhor organização dos módulos
- Adicionado ficheiro README com instruções de instalação e execução
- Correção de configuração de ambiente Python
- Melhoria dos comentários para a parte 1.2

## [v1.4] - 13 de outubro de 2023

**Contexto**: Estudo de redes neuronais artificiais

**Objectivo**: Resolução do problema de classificação dos padrões A (caixa) e B (cruz)

**Requisitos**:
- A rede deve ser representada com base numa classe designada RedeNeuronal
- A implementação deve ser realizada na linguagem Python
- A implementação deve ser realizada de acordo com o estudado nas aulas

## [v1.5] - 20 de outubro de 2023

## [v1.6] - 27 de outubro de 2023

## [v1.6] - 27 de outubro de 2023

## [v1.7] - 27 de outubro de 2023

## [v1.8] - 3 de novembro de 2023

## [v1.9] - 10 de novembro de 2023

## [v1.10] - 17 de novembro de 2023

## [v1.11] - 24 de novembro de 2023

## [v1.12] - 1 de dezembro de 2023

## [v1.13] - 8 de dezembro de 2023

### [v1.14] - 15 de dezembro de 2023

- Entrega do relatório final

## Bibliografia

- S. Russell, P. Norvig - [Artificial Intelligence: A Modern Approach](https://aima.cs.berkeley.edu/global-index.html), 4th Global Edition, Prentice Hall, 2022

- R. Sutton, A Barto - [Reinforcement Learning: An Introduction](https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf), 2nd Edition, MIT Press, 2020

- C. Aggarwal - [Neural Networks and Deep Learning](https://www.charuaggarwal.net/neural.htm), Springer, 2018

- Michael Wooldridge - [An Introduction to Multi-Agent Systems](https://www.cs.ox.ac.uk/people/michael.wooldridge/pubs/imas/IMAS2e.html), John Wiley & Sons, 2009

- Rolf Pfeifer, Christian Scheier - [Understanding Intelligence](https://mitpress.mit.edu/9780262661256/understanding-intelligence/), MIT Press, 2001
