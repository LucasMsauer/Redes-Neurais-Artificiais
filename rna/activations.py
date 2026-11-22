"""Funções de ativação e suas derivadas.

Cada derivada recebe (z, a): z = soma ponderada, a = f(z). Assim, sigmoid e tanh
reaproveitam a saída já calculada.
"""
import math


def _linear(z):
    return z


def _d_linear(z, a):
    return 1.0


def _sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)  # evita overflow para z muito negativo
    return e / (1.0 + e)


def _d_sigmoid(z, a):
    return a * (1.0 - a)


def _tanh(z):
    return math.tanh(z)


def _d_tanh(z, a):
    return 1.0 - a * a


def _relu(z):
    return z if z > 0.0 else 0.0


def _d_relu(z, a):
    return 1.0 if z > 0.0 else 0.0


ACTIVATIONS = {
    "linear": (_linear, _d_linear),
    "sigmoid": (_sigmoid, _d_sigmoid),
    "tanh": (_tanh, _d_tanh),
    "relu": (_relu, _d_relu),
}


def get_activation(name):
    key = name.strip().lower()
    if key not in ACTIVATIONS:
        raise ValueError(
            f"Função de ativação desconhecida: '{name}'. Opções: {', '.join(ACTIVATIONS)}"
        )
    return key, *ACTIVATIONS[key]
