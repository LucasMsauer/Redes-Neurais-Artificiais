# Trabalho Prático 02 – MLP configurável (Python, sem bibliotecas de ML)

Usa apenas a biblioteca padrão do Python (3.9+). Nenhuma dependência externa.

## Estrutura
| Arquivo | Responsabilidade (item do enunciado) |
|---|---|
| `rna/neuron.py` | Neurônio: pesos, viés, z, saída, delta (1.1) |
| `rna/layer.py` | Camada + inicialização Xavier/He (1.2, 1.5) |
| `rna/network.py` | Rede: forward, MSE, backpropagation, SGD (1.3, 1.6, 1.7, 1.8) |
| `rna/activations.py` | Linear, Sigmoid, Tanh, ReLU + derivadas (1.4) |
| `rna/dataset.py` | CSV, divisão treino/val/teste, normalização (1.9) |
| `rna/trainer.py` | Laço de treinamento SGD, MSE por época (1.10) |
| `rna/persistence.py` | Salvar/carregar JSON (1.11) |
| `rna/predictor.py` | Previsão (1.12) |
| `main.py` | Interface de linha de comando |
| `tests/test_rna.py` | Testes (checagem numérica do gradiente, salvar/carregar, seed...) |

## Uso
```bash
# treinar (arquitetura definida na linha de comando: neurônios:ativação, a última é a saída)
python main.py train --csv dataset.csv --layers 10:sigmoid,1:linear --lr 0.0005 --epochs 4000 \
    --split 70,15,15 --seed 1 --normalize zscore --bias-from-targets \
    --out modelo.json --history historico.csv

python main.py info --model modelo.json
python main.py predict --model modelo.json --values 150,30     # sementes_m2, nitrogenio_kg_ha

python -m unittest discover -s tests -v
```
Opções úteis: `--inputs`/`--outputs` (nomes ou índices de colunas), `--normalize none|minmax|zscore`,
`--lr-decay` (decaimento opcional da taxa; continua sendo SGD).

## Decisões de projeto
- **Erro por exemplo:** E = (1/m)·Σ(a−t)². Delta de saída = (2/m)(a−t)·f'(z). MSE do conjunto = média dos E.
- **Backprop:** todos os deltas são calculados *antes* de qualquer atualização (usa os pesos do forward).
- **Inicialização:** He (normal) para ReLU; Xavier/Glorot (uniforme) para Linear/Sigmoid/Tanh. Viés começa em 0.
- **Normalização:** só nas entradas; parâmetros calculados apenas no treino, aplicados aos 3 conjuntos e salvos no JSON
  (o `predict` reaplica automaticamente).
- **`--bias-from-targets` (opcional):** como as saídas não são normalizadas (~2500), o viés inicial da saída recebe a média
  dos alvos de treino, para a rede não gastar épocas só para chegar nessa escala.
- **Dataset fornecido:** 179 linhas, 2 entradas (`sementes_m2`, `nitrogenio_kg_ha`) e 1 saída (`produtividade_kg_ha`).
  Os dados são ruidosos: a variância total é ~61 mil e a variância intra-grupo (piso aproximado do MSE) ~20 mil.

## Interface gráfica (Streamlit)
```bash
pip install -r requirements.txt
streamlit run app.py
```
Abas: configurar/treinar (com gráfico de MSE ao vivo), prever e carregar rede salva. Usa o mesmo pacote `rna/` da linha de comando.
