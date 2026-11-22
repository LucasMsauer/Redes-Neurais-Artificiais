# Sistema Configurável de Redes Neurais Artificiais (MLP)

Trabalho Prático 02 – Inteligência Artificial.

Aplicação que **constrói, configura, treina, salva e utiliza** redes neurais do tipo
*Multilayer Perceptron* (MLP). A arquitetura (quantas camadas, quantos neurônios,
qual função de ativação em cada camada) é definida pelo usuário, **sem alterar o código-fonte**.

Todos os algoritmos (propagação direta, erro, retropropagação e SGD) foram implementados do zero,
em Python puro. Não há TensorFlow, Keras, PyTorch nem NumPy no núcleo da rede.

---

## Sumário
1. [Visão geral](#1-visão-geral)
2. [Instalação](#2-instalação)
3. [Como usar](#3-como-usar)
4. [Como a rede funciona](#4-como-a-rede-funciona)
5. [Estrutura do projeto](#5-estrutura-do-projeto)
6. [Requisitos do enunciado × onde estão](#6-requisitos-do-enunciado--onde-estão)
7. [Formato do arquivo salvo (JSON)](#7-formato-do-arquivo-salvo-json)
8. [O dataset fornecido](#8-o-dataset-fornecido)
9. [Decisões de projeto](#9-decisões-de-projeto)
10. [Testes](#10-testes)
11. [Roteiro sugerido para a demonstração](#11-roteiro-sugerido-para-a-demonstração)
12. [Perguntas frequentes](#12-perguntas-frequentes)

---

## 1. Visão geral

O sistema tem duas formas de uso, que compartilham o **mesmo núcleo** (`rna/`):

| Interface | Arquivo | Para que serve |
|---|---|---|
| Linha de comando | `main.py` | Treinar, inspecionar e prever por comandos |
| Interface gráfica (Streamlit) | `app.py` | Fazer o mesmo de forma visual, com gráfico de erro ao vivo |

Fluxo típico:

```
CSV ──► escolher colunas ──► dividir (treino/validação/teste) ──► normalizar (opcional)
    ──► treinar com SGD ──► avaliar no teste ──► salvar JSON ──► carregar ──► prever
```

---

## 2. Instalação

Requisito: **Python 3.9 ou mais novo**. O núcleo e a linha de comando não precisam de nada além do Python.

Para a interface gráfica:

```bash
pip install -r requirements.txt     # instala o Streamlit
```

No Windows, se o comando `streamlit` não for reconhecido, use `python -m streamlit ...`.

---

## 3. Como usar

### 3.1 Interface gráfica

```bash
python -m streamlit run app.py
```

O navegador abre em `http://localhost:8501`, com três abas:

- **Configurar e treinar:** escolha o CSV, as colunas de entrada e saída, a arquitetura
  (camadas ocultas, neurônios e ativação de cada uma), taxa de aprendizado, épocas, seed,
  normalização e divisão treino/validação/teste. Durante o treino aparece o gráfico do MSE
  de treino e de validação. No final aparecem o MSE de teste e um gráfico de
  **valor previsto × valor real**. Há também o botão para baixar a rede em JSON.
- **Prever:** informe os valores de entrada (sem normalizar). O app mostra a faixa de valores vista
  no treino e avisa se você digitar algo fora dela.
- **Carregar rede:** envie um JSON salvo e a rede é reconstruída, pronta para prever.

### 3.2 Linha de comando

**Treinar** (a arquitetura vem do argumento `--layers`, no formato `neurônios:ativação`, e a
última camada é a de saída):

```bash
python main.py train --csv dataset.csv --layers 10:sigmoid,1:linear --lr 0.0005 --epochs 4000 --split 70,15,15 --seed 1 --normalize zscore --bias-from-targets --out modelo.json --history historico.csv
```

**Ver a arquitetura de uma rede salva:**

```bash
python main.py info --model modelo.json
```

**Prever** (valores na ordem das colunas de entrada: `sementes_m2`, `nitrogenio_kg_ha`):

```bash
python main.py predict --model modelo.json --values 250,30
```

Parâmetros do `train`:

| Parâmetro | Significado | Padrão |
|---|---|---|
| `--csv` | arquivo de dados | (obrigatório) |
| `--inputs` / `--outputs` | colunas de entrada/saída (nomes ou índices, separados por vírgula) | saída = última coluna, entradas = o resto |
| `--layers` | arquitetura, ex.: `8:tanh,4:relu,1:linear` | (obrigatório) |
| `--lr` | taxa de aprendizado | 0.001 |
| `--epochs` | número de épocas | 300 |
| `--split` | % treino,validação,teste (soma 100) | 70,15,15 |
| `--seed` | semente (inicialização, divisão e embaralhamento) | aleatória |
| `--normalize` | `none`, `minmax` ou `zscore` (só nas entradas) | none |
| `--lr-decay` | decaimento opcional da taxa: `lr / (1 + decay·(época−1))` | 0 (sem decaimento) |
| `--bias-from-targets` | viés inicial da saída = média das saídas de treino | desligado |
| `--out` | arquivo da rede salva | modelo.json |
| `--history` | CSV com o MSE por época | (não salva) |

Ativações disponíveis: `linear`, `sigmoid`, `tanh`, `relu`.

---

## 4. Como a rede funciona

### 4.1 Neurônio
Cada neurônio guarda seus **pesos** e seu **viés** (*bias*). Dada uma lista de entradas `x`:

```
z      = viés + Σ (peso_i · x_i)        (soma ponderada)
saída  = f(z)                           (f = função de ativação)
```

Para o treinamento, o neurônio também guarda `z`, a `saída` e o **delta** `δ = ∂E/∂z`.

### 4.2 Camada e rede
Uma **camada** é um grupo de neurônios que recebem as mesmas entradas, com uma função de
ativação própria. A **rede** é uma sequência de camadas: a saída de uma é a entrada da seguinte.
Camadas diferentes podem usar ativações diferentes.

### 4.3 Funções de ativação

| Função | f(z) | f'(z) |
|---|---|---|
| Linear | z | 1 |
| Sigmoid | 1 / (1 + e⁻ᶻ) | f · (1 − f) |
| Tanh | tanh(z) | 1 − f² |
| ReLU | max(0, z) | 1 se z > 0, senão 0 |

### 4.4 Propagação direta (*forward*)
As entradas passam camada por camada até a saída. O algoritmo é um laço sobre `net.layers`, então
funciona com qualquer número de camadas e neurônios.

### 4.5 Erro (MSE)
Para um exemplo com `m` saídas:

```
E = (1/m) · Σ (saída − esperado)²
```

O **MSE de um conjunto** é a média de E sobre todos os seus exemplos. É calculado separadamente para
treino, validação e teste.

### 4.6 Retropropagação (*backpropagation*)
Usa a regra da cadeia para descobrir quanto cada neurônio contribuiu para o erro:

```
Camada de saída:   δ_k = (2/m) · (saída_k − esperado_k) · f'(z_k)
Camadas ocultas:   δ_j = f'(z_j) · Σ_k ( peso_kj · δ_k )     (k = neurônios da camada seguinte)
```

Depois, **com todos os deltas já calculados**, os parâmetros são atualizados (descida do gradiente):

```
peso_ji ← peso_ji − taxa · δ_j · entrada_i
viés_j  ← viés_j  − taxa · δ_j
```

Atenção: todos os deltas são calculados *antes* de qualquer atualização. Assim, o delta de uma camada
oculta usa os mesmos pesos da propagação direta daquele exemplo (exigência do item 1.8).

### 4.7 Treinamento com SGD
No *Stochastic Gradient Descent*, os pesos são atualizados **após cada exemplo** de treino. A cada época:

1. A ordem dos exemplos de treino é embaralhada (reprodutível pela seed).
2. Para cada exemplo: forward → erro → backpropagation → atualização.
3. Ao final, calculam-se o MSE de treino e o MSE de validação.

O conjunto de **validação** só mede o erro (serve para acompanhar se a rede generaliza) e nunca altera os
pesos. O conjunto de **teste** só é usado na avaliação final.

### 4.8 Inicialização dos pesos
- **ReLU:** inicialização *He* (distribuição normal com desvio √(2 / entradas)).
- **Linear, Sigmoid, Tanh:** inicialização *Xavier/Glorot* (distribuição uniforme em ±√(6 / (entradas + saídas))).
- **Vieses:** começam em 0 (e são salvos, como os pesos).
- A **seed** torna a inicialização reproduzível.

### 4.9 Divisão e normalização dos dados
- A divisão é aleatória, com percentuais configuráveis (somando 100) e seed.
- A normalização (Min-Max ou Z-score) é feita **depois** da divisão, com parâmetros calculados
  **somente no conjunto de treino**, e aplicada aos três conjuntos. Assim, validação e teste não influenciam o treino.
- Só as **entradas** são normalizadas; as saídas esperadas não.
- Os parâmetros ficam salvos no JSON e são reaplicados automaticamente na previsão.

---

## 5. Estrutura do projeto

```
.
├── main.py              # interface de linha de comando
├── app.py               # interface gráfica (Streamlit)
├── dataset.csv          # dados fornecidos pelo professor
├── requirements.txt     # dependência da interface gráfica
├── modelo_exemplo.json  # exemplo de rede treinada e salva
├── historico.csv        # exemplo de MSE por época
├── rna/                 # núcleo do sistema
│   ├── neuron.py        # entidade Neurônio
│   ├── layer.py         # entidade Camada + inicialização Xavier/He
│   ├── network.py       # entidade Rede: forward, MSE, backpropagation
│   ├── activations.py   # Linear, Sigmoid, Tanh, ReLU e derivadas
│   ├── dataset.py       # leitura de CSV, divisão, normalização
│   ├── trainer.py       # laço de treinamento SGD
│   ├── persistence.py   # salvar/carregar JSON
│   └── predictor.py     # previsão com rede treinada
└── tests/
    └── test_rna.py      # testes automatizados
```

A separação segue o enunciado: entidades (`neuron`, `layer`, `network`), algoritmos de treino
(`trainer`), dados (`dataset`) e persistência (`persistence`) ficam em módulos distintos.
As duas interfaces apenas chamam esses módulos.

---

## 6. Requisitos do enunciado × onde estão

| Item | Requisito | Onde |
|---|---|---|
| 1.1 | Entidade Neurônio (pesos, viés, z, saída, delta) | `rna/neuron.py` |
| 1.2 | Entidade Camada | `rna/layer.py` |
| 1.3 | Entidade Rede Neural, composição dinâmica | `rna/network.py` (`NeuralNetwork.build`) |
| 1.4 | Linear, Sigmoid, Tanh, ReLU + derivadas, ativação por camada | `rna/activations.py` |
| 1.5 | Xavier/He, seed, viés armazenado | `rna/layer.py` |
| 1.6 | Propagação direta | `NeuralNetwork.forward` |
| 1.7 | MSE de treino, validação e teste | `NeuralNetwork.mse` |
| 1.8 | Backpropagation (deltas antes da atualização) | `NeuralNetwork.backward` |
| 1.9 | CSV, divisão com seed, normalização opcional só com o treino | `rna/dataset.py` |
| 1.10 | Treinamento SGD, MSE por época | `rna/trainer.py` |
| 1.11 | Salvar e carregar em JSON (inclui normalização) | `rna/persistence.py` |
| 1.12 | Previsão (só forward, valida nº de entradas, aplica normalização) | `rna/predictor.py` |
| Restrição 4 | Arquitetura não fixa no código | `--layers` / telas do Streamlit |
| Restrição 5 | Interface de linha de comando e gráfica | `main.py`, `app.py` |
| Restrição 6 | Código modular | pacote `rna/` |

---

## 7. Formato do arquivo salvo (JSON)

```json
{
  "format": "mlp-json-v1",
  "network": {
    "n_inputs": 2,
    "layers": [
      {
        "activation": "sigmoid",
        "neurons": [ { "weights": [0.12, -0.5], "bias": 0.03 }, "..." ]
      },
      {
        "activation": "linear",
        "neurons": [ { "weights": ["..."], "bias": 2750.1 } ]
      }
    ]
  },
  "normalization": {
    "method": "zscore",
    "params": { "mean": [300.0, 60.0], "std": [111.8, 42.4] }
  },
  "meta": {
    "input_names": ["sementes_m2", "nitrogenio_kg_ha"],
    "output_names": ["produtividade_kg_ha"],
    "input_ranges": [[150, 450], [0, 120]],
    "lr": 0.0005, "epochs": 4000, "seed": 1
  }
}
```

Contém tudo o que o item 1.11 exige: entradas, camadas e neurônios, ativação de cada camada,
pesos, vieses e a normalização com seus parâmetros. O campo `meta` guarda informações auxiliares
(nomes das colunas, hiperparâmetros), mas não é necessário para reconstruir a rede.
Uma rede salva e carregada produz **exatamente** as mesmas saídas (verificado em teste automatizado).

---

## 8. O dataset fornecido

`dataset.csv` tem 179 linhas, com 2 entradas e 1 saída (problema de **regressão**):

| Coluna | Papel | Valores |
|---|---|---|
| `sementes_m2` | entrada | 150, 250, 350, 450 |
| `nitrogenio_kg_ha` | entrada | 0, 30, 60, 90, 120 |
| `produtividade_kg_ha` | saída | de 1700,9 a 3160,4 |

Como o problema é de regressão, a camada de saída usa ativação **linear**, com 1 neurônio.

**Sobre o erro:** os dados são ruidosos. Repetições com as mesmas entradas dão saídas diferentes. A
variância total das saídas é de cerca de 61 mil, e a variância dentro de cada grupo de entradas
(um piso aproximado para o MSE) é de cerca de 20 mil. Por isso o MSE nunca chega perto de zero, e o MSE de teste
varia bastante conforme a seed da divisão, já que o teste tem poucas amostras (27 com 70/15/15).

---

## 9. Decisões de projeto

- **Linguagem e dependências:** Python puro no núcleo. O Streamlit é usado só na interface.
- **Erro por exemplo:** `E = (1/m) Σ (a − t)²`, com o gradiente correspondente `(2/m)(a − t)`.
- **Opções extras (opcionais, desligadas por padrão):**
  - `--bias-from-targets`: as saídas não podem ser normalizadas (enunciado) e valem cerca de 2500.
    Sem ajuda, a rede gasta muitas épocas só para levar o viés da saída até essa escala. Esta opção define o
    viés inicial da saída como a média das saídas de **treino** (não usa validação nem teste).
  - `--lr-decay`: reduz a taxa de aprendizado ao longo das épocas. Continua sendo SGD (atualização por exemplo).
- **Embaralhamento:** a ordem dos exemplos muda a cada época, reprodutível pela seed.
- **Normalização de valores fora da faixa:** valores de previsão fora do intervalo visto no treino são
  transformados normalmente (sem cortar). A interface gráfica apenas avisa.

---

## 10. Testes

```bash
python -m unittest discover -s tests -v
```

| Teste | O que verifica |
|---|---|
| `test_gradiente_numerico` | o backpropagation bate com a derivada calculada por diferenças finitas, em uma rede com as 4 ativações |
| `test_salvar_carregar` | rede salva e recarregada dá exatamente as mesmas saídas, com normalização |
| `test_seed_reprodutivel` | mesma seed gera os mesmos pesos iniciais |
| `test_divisao` | divisão reprodutível, tamanhos corretos, erro se os percentuais não somam 100 |
| `test_normalizacao_so_treino` | parâmetros de normalização vêm só do treino |
| `test_validacao_entradas` | previsão com número errado de entradas gera erro |
| `test_predict_nao_altera_pesos` | prever não modifica a rede |

---

## 11. Roteiro sugerido para a demonstração

1. **Estrutura do código:** mostrar o pacote `rna/` e a responsabilidade de cada arquivo (seção 5).
2. **Algoritmos principais:** explicar `forward`, `backward` (deltas antes da atualização), a inicialização e a
   normalização só com o treino.
3. **Teste do gradiente:** rodar os testes e explicar que o backpropagation foi validado numericamente.
4. **Treinamento:** na interface gráfica, configurar uma arquitetura, treinar e mostrar o gráfico do MSE e o
   gráfico previsto × real.
5. **Persistência:** fazer uma previsão (ex.: sementes 250, nitrogênio 30), anotar o valor, baixar o JSON,
   recarregar a página, carregar o JSON e repetir a previsão: o valor deve ser idêntico.
6. **Flexibilidade:** mudar a arquitetura (ex.: duas camadas ocultas, tanh e relu) sem mexer no código.
7. **Linha de comando (opcional):** mostrar `train`, `info` e `predict`.

---

## 12. Perguntas frequentes

**Por que o MSE de teste é diferente do de treino?**
Porque o teste tem poucas amostras e os dados são ruidosos. Mudar a seed muda quais amostras caem no teste.

**Por que a rede não chega a MSE próximo de zero?**
O ruído do dataset impede (seção 8). O objetivo é chegar perto do piso, não de zero.

**Posso usar mais de uma saída?**
Sim. Escolha mais colunas de saída. A camada final passa a ter um neurônio por saída.

**Posso usar outro CSV?**
Sim, desde que seja numérico e já limpo. Escolha as colunas de entrada e saída.

**O que acontece se eu digitar o número errado de entradas na previsão?**
O sistema recusa e informa quantos valores a rede espera.
