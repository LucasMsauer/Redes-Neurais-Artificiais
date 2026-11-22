"""Interface de linha de comando do sistema de MLP.

Exemplos:
  python main.py train --csv dataset.csv --layers 8:tanh,4:tanh,1:linear \
      --lr 0.001 --epochs 300 --split 70,15,15 --seed 42 --normalize minmax --out modelo.json
  python main.py predict --model modelo.json --values 150,30
  python main.py info --model modelo.json
"""
import argparse
import csv
import sys

from rna import dataset as ds
from rna.network import NeuralNetwork
from rna.persistence import load_model, save_model
from rna.predictor import predict
from rna.trainer import train


def parse_layers(text):
    specs = []
    for part in text.split(","):
        try:
            n, act = part.split(":")
            specs.append((int(n), act.strip().lower()))
        except ValueError:
            raise SystemExit(f"Camada inválida '{part}'. Use o formato neurônios:ativação (ex.: 8:tanh).")
    return specs


def csv_list(text):
    return [t for t in text.split(",")] if text else None


def cmd_train(a):
    X, Y, in_names, out_names = ds.load_csv(a.csv, csv_list(a.inputs), csv_list(a.outputs))
    specs = parse_layers(a.layers)
    net = NeuralNetwork.build(len(in_names), specs, seed=a.seed)
    ds.check_compatibility(net, X, Y)

    pct = [float(p) for p in a.split.split(",")]
    if len(pct) != 3:
        raise SystemExit("--split precisa de 3 valores: treino,validacao,teste")
    train_set, val_set, test_set = ds.split(X, Y, *pct, seed=a.seed)

    normalizer = None
    if a.normalize != "none":
        normalizer = ds.Normalizer.fit(a.normalize, train_set[0])  # só com o treino
        train_set = (normalizer.transform(train_set[0]), train_set[1])
        val_set = (normalizer.transform(val_set[0]), val_set[1])
        test_set = (normalizer.transform(test_set[0]), test_set[1])

    if a.bias_from_targets:  # viés de saída inicial = média dos alvos de TREINO
        cols = list(zip(*train_set[1]))
        net.set_output_bias([sum(c) / len(c) for c in cols])

    print(f"Entradas: {in_names} | Saídas: {out_names}")
    print(f"Amostras -> treino: {len(train_set[0])}, validação: {len(val_set[0])}, teste: {len(test_set[0])}")

    def show(rec):
        if rec["epoch"] == 1 or rec["epoch"] % a.print_every == 0 or rec["epoch"] == a.epochs:
            print(f"época {rec['epoch']:5d} | MSE treino {rec['train_mse']:12.4f} | MSE validação {rec['val_mse']:12.4f}")

    history = train(net, train_set, val_set, a.epochs, a.lr, seed=a.seed, on_epoch=show, lr_decay=a.lr_decay)
    print(f"\nMSE de teste (avaliação final): {net.mse(*test_set):.4f}")

    meta = {"input_names": in_names, "output_names": out_names, "lr": a.lr, "lr_decay": a.lr_decay, "epochs": a.epochs, "seed": a.seed}
    save_model(a.out, net, normalizer, meta)
    print(f"Rede salva em {a.out}")
    if a.history:
        with open(a.history, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["epoch", "train_mse", "val_mse"])
            w.writeheader()
            w.writerows(history)
        print(f"Histórico salvo em {a.history}")


def cmd_predict(a):
    net, norm, meta = load_model(a.model)
    values = [float(v) for v in a.values.split(",")]
    out = predict(net, norm, values)
    names = meta.get("output_names") or [f"saida_{i}" for i in range(len(out))]
    for n, v in zip(names, out):
        print(f"{n}: {v:.4f}")


def cmd_info(a):
    net, norm, meta = load_model(a.model)
    print(f"Entradas: {net.n_inputs} {meta.get('input_names', '')}")
    for i, l in enumerate(net.layers, 1):
        print(f"Camada {i}: {len(l.neurons)} neurônios, ativação {l.activation_name}")
    print(f"Normalização: {norm.method if norm else 'nenhuma'}")


def main():
    p = argparse.ArgumentParser(description="MLP configurável")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="criar, treinar e salvar uma rede")
    t.add_argument("--csv", required=True)
    t.add_argument("--inputs", help="colunas de entrada (nomes ou índices, separados por vírgula)")
    t.add_argument("--outputs", help="colunas de saída (padrão: última coluna)")
    t.add_argument("--layers", required=True, help="ex.: 8:tanh,4:relu,1:linear (a última é a saída)")
    t.add_argument("--lr", type=float, default=0.001)
    t.add_argument("--lr-decay", type=float, default=0.0, help="decaimento da taxa: lr/(1+decay*(época-1))")
    t.add_argument("--bias-from-targets", action="store_true", help="viés de saída inicial = média das saídas de treino")
    t.add_argument("--epochs", type=int, default=300)
    t.add_argument("--split", default="70,15,15", help="treino,validação,teste (soma 100)")
    t.add_argument("--seed", type=int, default=None)
    t.add_argument("--normalize", choices=["none", "minmax", "zscore"], default="none")
    t.add_argument("--out", default="modelo.json")
    t.add_argument("--history", help="CSV para salvar MSE por época")
    t.add_argument("--print-every", type=int, default=10)
    t.set_defaults(func=cmd_train)

    q = sub.add_parser("predict", help="prever com uma rede salva")
    q.add_argument("--model", required=True)
    q.add_argument("--values", required=True, help="valores de entrada separados por vírgula")
    q.set_defaults(func=cmd_predict)

    i = sub.add_parser("info", help="mostrar a arquitetura de uma rede salva")
    i.add_argument("--model", required=True)
    i.set_defaults(func=cmd_info)

    a = p.parse_args()
    try:
        a.func(a)
    except (ValueError, FileNotFoundError) as e:
        sys.exit(f"Erro: {e}")


if __name__ == "__main__":
    main()
