"""Salvar e carregar a rede (JSON)."""
import json

from .dataset import Normalizer
from .network import NeuralNetwork


def save_model(path, net, normalizer=None, meta=None):
    data = {
        "format": "mlp-json-v1",
        "network": net.to_dict(),
        "normalization": normalizer.to_dict() if normalizer else None,
        "meta": meta or {},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)  # floats do Python são ida-e-volta exatos


def load_model(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    net = NeuralNetwork.from_dict(data["network"])
    norm = Normalizer.from_dict(data.get("normalization"))
    return net, norm, data.get("meta", {})
