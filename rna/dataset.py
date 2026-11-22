"""Leitura de CSV, divisão treino/validação/teste e normalização."""
import csv
import random


def _resolve(spec, header):
    """Aceita nomes de coluna ou índices (base 0)."""
    cols = []
    for item in spec:
        item = item.strip()
        if item in header:
            cols.append(header.index(item))
        elif item.lstrip("-").isdigit() and -len(header) <= int(item) < len(header):
            cols.append(int(item) % len(header))
        else:
            raise ValueError(f"Coluna '{item}' não encontrada. Colunas: {header}")
    return cols


def load_csv(path, input_cols=None, output_cols=None, delimiter=","):
    """Retorna (X, Y, nomes_entradas, nomes_saidas).

    Padrão: a última coluna é a saída e todas as outras são entradas.
    """
    with open(path, newline="", encoding="utf-8-sig") as f:  # utf-8-sig remove BOM
        rows = [r for r in csv.reader(f, delimiter=delimiter) if r]
    header = [h.strip() for h in rows[0]]
    outs = _resolve(output_cols, header) if output_cols else [len(header) - 1]
    ins = _resolve(input_cols, header) if input_cols else [i for i in range(len(header)) if i not in outs]
    if set(ins) & set(outs):
        raise ValueError("Uma coluna não pode ser entrada e saída ao mesmo tempo.")
    X, Y = [], []
    for n, r in enumerate(rows[1:], start=2):
        try:
            X.append([float(r[i]) for i in ins])
            Y.append([float(r[i]) for i in outs])
        except (ValueError, IndexError):
            raise ValueError(f"Linha {n} do CSV inválida: {r}")
    return X, Y, [header[i] for i in ins], [header[i] for i in outs]


def check_compatibility(net, X, Y):
    if len(X[0]) != net.n_inputs:
        raise ValueError(f"Dataset tem {len(X[0])} entradas, mas a rede espera {net.n_inputs}.")
    if len(Y[0]) != net.n_outputs:
        raise ValueError(f"Dataset tem {len(Y[0])} saídas, mas a rede tem {net.n_outputs} neurônios de saída.")


def split(X, Y, train_pct, val_pct, test_pct, seed=None):
    """Divisão aleatória e reprodutível. Os percentuais devem somar 100."""
    if abs(train_pct + val_pct + test_pct - 100) > 1e-9 or min(train_pct, val_pct, test_pct) < 0:
        raise ValueError("Os percentuais de treino/validação/teste devem somar 100.")
    idx = list(range(len(X)))
    random.Random(seed).shuffle(idx)
    n = len(idx)
    n_train = round(n * train_pct / 100)
    n_val = round(n * val_pct / 100)
    parts = (idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:])
    return tuple(([X[i] for i in p], [Y[i] for i in p]) for p in parts)


class Normalizer:
    """Normalização das ENTRADAS. Parâmetros calculados só no conjunto de treino."""

    METHODS = ("minmax", "zscore")

    def __init__(self, method, params):
        if method not in self.METHODS:
            raise ValueError(f"Normalização desconhecida: {method}. Opções: {self.METHODS}")
        self.method = method
        self.params = params  # minmax: {"min","max"}; zscore: {"mean","std"}

    @classmethod
    def fit(cls, method, X_train):
        cols = list(zip(*X_train))
        if method == "minmax":
            return cls(method, {"min": [min(c) for c in cols], "max": [max(c) for c in cols]})
        if method == "zscore":
            mean = [sum(c) / len(c) for c in cols]
            std = [(sum((v - m) ** 2 for v in c) / len(c)) ** 0.5 for c, m in zip(cols, mean)]
            return cls(method, {"mean": mean, "std": std})
        raise ValueError(f"Normalização desconhecida: {method}")

    def transform_row(self, x):
        if self.method == "minmax":
            lo, hi = self.params["min"], self.params["max"]
            return [(v - a) / (b - a) if b != a else 0.0 for v, a, b in zip(x, lo, hi)]
        mean, std = self.params["mean"], self.params["std"]
        return [(v - m) / s if s != 0 else 0.0 for v, m, s in zip(x, mean, std)]

    def transform(self, X):
        return [self.transform_row(x) for x in X]

    def to_dict(self):
        return {"method": self.method, "params": self.params}

    @classmethod
    def from_dict(cls, d):
        return cls(d["method"], d["params"]) if d else None
